import os
import gc
import lpips
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from fidesr import CSDLoss, FiDeSR
from src.my_utils.training_utils import parse_args  
from src.datasets.dataset import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

_KERNEL_CACHE = {}

def get_kernel(name, dtype, device):
    key = f"{name}_{dtype}_{device}"
    if key not in _KERNEL_CACHE:
        if name == "sobel_x":
            kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1,1,3,3)
        elif name == "sobel_y":
            kernel = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=dtype, device=device).view(1,1,3,3)
        elif name == "laplacian":
            kernel = torch.tensor([[0, 1, 0], [1,-4, 1], [0, 1, 0]], dtype=dtype, device=device).view(1,1,3,3)
        elif name == "box3":
            kernel = torch.ones(1,1,3,3, device=device, dtype=dtype) / 9.0
        else:
            raise ValueError(f"Unknown kernel: {name}")
        _KERNEL_CACHE[key] = kernel
    return _KERNEL_CACHE[key]

def conv3x3_same_replicate(x, weight):
    """
    Helper function for 3x3 convolution with replicate padding
    weight: [out_c, in_c, 3, 3], x: [B, C, H, W]
    """
    x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
    weight = weight.to(dtype=x.dtype, device=x.device)
    return F.conv2d(x_pad, weight, padding=0)

def build_D(y):
    """Build detail map D from ground-truth image y.

    D highlights local structure by combining:
    1) Sobel gradient magnitude (edge strength),
    2) absolute Laplacian response (high-frequency components),
    3) local variance (texture/roughness).
    Then it applies robust per-image quantile normalization (5-95%).
    """
    y = y.to(dtype=torch.float32)
    if y.shape[1] == 3:
        y_gray = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    else:
        y_gray = y.to(dtype=torch.float32)
    
    device, dtype =y_gray.device,y_gray.dtype
    sobel_x_kernel = get_kernel("sobel_x", dtype, device)
    sobel_y_kernel = get_kernel("sobel_y", dtype, device)
    laplacian_kernel = get_kernel("laplacian", dtype, device)
    box3_kernel = get_kernel("box3", dtype, device)

    grad_x = conv3x3_same_replicate(y_gray, sobel_x_kernel)
    grad_y = conv3x3_same_replicate(y_gray, sobel_y_kernel)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    lap_response = conv3x3_same_replicate(y_gray, laplacian_kernel).abs()
    local_mean = conv3x3_same_replicate(y_gray, box3_kernel)
    local_var = conv3x3_same_replicate((y_gray - local_mean) ** 2, box3_kernel)

    # Combine edge + high-frequency + local texture
    detail_map = (grad_magnitude + lap_response + local_var) / 3.0

    # Normalize to [0, 1] using percentile clipping
    batch_size = detail_map.shape[0]
    flat_map = detail_map.view(batch_size, -1)
    q05 = torch.quantile(flat_map, 0.05, dim=1, keepdim=True).view(batch_size, 1, 1, 1)
    q95 = torch.quantile(flat_map, 0.95, dim=1, keepdim=True).view(batch_size, 1, 1, 1)
    detail_map = torch.clamp((detail_map - q05) / (q95 - q05 + 1e-6), 0, 1)

    detail_map = conv3x3_same_replicate(detail_map, box3_kernel)
    return detail_map

def build_E(y_hat, y, lpips_model, p=0.5):
    """Build error map E from predicted image y_hat and ground-truth image y.

    E combines:
    1) pixel-domain error (mean + std over channels),
    2) perceptual error from LPIPS.
    Then it applies robust per-image quantile normalization (5-95%).
    """
    y_hat = y_hat.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

     # Pixel error (detail-aware: channel mean + channel std)
    pixel_abs = torch.abs(y_hat - y)
    pixel_mean = pixel_abs.mean(dim=1, keepdim=True)
    pixel_std = pixel_abs.std(dim=1, keepdim=True)
    E_pix = 0.8 * pixel_mean + 0.2 * pixel_std
    
    # Perceptual error (LPIPS)
    with torch.no_grad():
        E_perc = lpips_model(y_hat, y)

    H, W = y.shape[-2], y.shape[-1]
    if E_perc.dim() == 4:
        if E_perc.shape[2] == 1 and E_perc.shape[3] == 1:
            E_perc = E_perc.expand(-1, -1, H, W)
        elif (E_perc.shape[2] != H) or (E_perc.shape[3] != W):
            E_perc = F.interpolate(E_perc, size=(H, W), mode="bilinear", align_corners=False)
    else:
        E_perc = E_perc.view(E_perc.shape[0], 1, 1, 1).expand(-1, -1, H, W)
    
    # Combine pixel + perceptual error
    error_map = (1 - p) * E_pix + p * E_perc
    error_map = error_map.to(dtype=torch.float32)
    
    # Normalize to [0, 1] using percentile clipping
    batch_size = error_map.shape[0]
    flat_map = error_map.view(batch_size, -1)
    q05 = torch.quantile(flat_map, 0.05, dim=1, keepdim=True).view(batch_size, 1, 1, 1)
    q95 = torch.quantile(flat_map, 0.95, dim=1, keepdim=True).view(batch_size, 1, 1, 1)
    error_map = torch.clamp((error_map - q05) / (q95 - q05 + 1e-6), 0, 1)
    
    return error_map

def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        broadcast_buffers=False, 
        bucket_cap_mb=25,
        static_graph=True 
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    fidesr = FiDeSR(args)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            fidesr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        fidesr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # init CSDLoss model
    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)

    net_lpips = lpips.LPIPS(net='vgg', spatial=True).cuda()
    net_lpips.requires_grad_(False)

    # # set gen adapter
    fidesr.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])
    fidesr.set_train()

    # make the optimizer
    layers_to_opt = []
    for n, _p in fidesr.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)
    for n, _p in fidesr.lrrb.named_parameters():
        layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    
    # initialize the dataset
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="test", args=args)
    
    # initialize DataLoader
    dl_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
        timeout=60,
        prefetch_factor=2
    )
    dl_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    

    # init RAM for text prompt extractor
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='src/ram_pretrain_model/ram_swin_large_14m.pth',
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    fidesr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        fidesr, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    lambda_l2 = args.lambda_l2
    lambda_lpips = args.lambda_lpips
    lambda_csd = args.lambda_csd
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(fidesr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                # get text prompts from GT
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{each_caption}, {args.pos_prompt_csd}' for each_caption in caption]
                    
                if args.is_module:
                    accelerator.unwrap_model(fidesr).set_train() 
                else:
                    fidesr.set_train()
                        
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = fidesr(x_src, x_tgt, batch=batch, args=args)

                # Initialize alpha values
                alpha_hf = 0.0
                alpha_lf = 0.0
                w_DAW = None

                # Difficulty Weighting Map
                if global_step >= args.daw_warmup_offset:
                    if args.daw_ramp_steps > 0:
                        t = min(global_step - args.daw_warmup_offset, args.daw_ramp_steps)
                        ramp = t / args.daw_ramp_steps
                    else:
                        ramp = 1.0
                    alpha_hf = ramp * args.alpha_daw_hf
                    alpha_lf = ramp * args.alpha_daw_lf

                    with torch.no_grad():
                        detail_map = build_D(x_tgt)
                        error_map = build_E(x_tgt_pred, x_tgt, net_lpips, p=args.daw_wperc)

                    W_hf = (detail_map * error_map) if alpha_hf > 0 else None

                    if alpha_lf > 0:
                        detail_map_hf = detail_map
                        for _ in range(args.daw_lf_smooth):
                            detail_map_hf = conv3x3_same_replicate(detail_map_hf, get_kernel("box3", detail_map_hf.dtype, detail_map_hf.device))
                        W_lf = (1.0 - detail_map_hf).clamp(0, 1) * error_map
                    else:
                        W_lf = None

                    def _soft_clip_and_norm(W):
                        blur_kernel = get_kernel("box3", W.dtype, W.device)
                        Wb = conv3x3_same_replicate(W, blur_kernel)
                        Wb = Wb / (Wb.mean(dim=(2,3), keepdim=True) + 1e-6)
                        return torch.tanh(Wb / args.wmax_daw) * args.wmax_daw

                    w = torch.ones_like(error_map)

                    if W_hf is not None:
                        w = w + alpha_hf * _soft_clip_and_norm(W_hf)
                    if W_lf is not None:
                        w = w + alpha_lf * _soft_clip_and_norm(W_lf)

                    w_mean = w.mean(dim=(2, 3), keepdim=True)
                    w_DAW = w / (w_mean + 1e-8)
                                        
                # L2 Loss with DAW weighting (MSE)
                if w_DAW is not None:
                    loss_l2 = (w_DAW * (x_tgt_pred.float() - x_tgt.float())**2).mean() * lambda_l2
                else:
                    loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l2

                # LPIPS Loss with DAW weighting (Perceptual)
                lpips_map = net_lpips(x_tgt_pred.float(), x_tgt.float())
                if w_DAW is not None:
                    if lpips_map.dim() == 4 and lpips_map.shape[-1] > 1:
                        w_resize = F.interpolate(w_DAW, size=lpips_map.shape[-2:], mode='bilinear', align_corners=False)
                        loss_lpips = (w_resize * lpips_map).mean() * lambda_lpips
                    else:
                        w_mean = w_DAW.mean(dim=(2, 3), keepdim=True)
                        loss_lpips = (w_mean * lpips_map).mean() * lambda_lpips
                else:
                    loss_lpips = lpips_map.mean() * lambda_lpips

                loss = loss_l2 + loss_lpips

                # CSD Loss with DAW weighting
                loss_csd = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args, weight_map=w_DAW) * lambda_csd
                loss = loss + loss_csd
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    all_params = []
                    for g in optimizer.param_groups:
                        all_params += g["params"]
                    accelerator.clip_grad_norm_(all_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_csd"] = loss_csd.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    
                    # log DAW metrics
                    logs["daw_alpha_hf"] = alpha_hf
                    logs["daw_alpha_lf"] = alpha_lf
                    
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(fidesr).save_model(outf)

                    # test
                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # get text prompts from LR
                                x_src_ram = ram_transforms(x_src * 0.5 + 0.5)
                                caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                batch_val["prompt"] = caption
                                # forward pass
                                x_tgt_pred, latents_pred, _, _ = accelerator.unwrap_model(fidesr)(x_src, x_tgt,
                                                                                                      batch=batch_val,
                                                                                                      args=args)
                                # save the output
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                else:
                                    pass
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)

                    accelerator.log(logs, step=global_step)

if __name__ == "__main__":
    args = parse_args()
    main(args)
