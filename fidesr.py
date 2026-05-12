import os
import sys
import time
import random
from types import SimpleNamespace
import glob
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook

def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# LRRB (Latent Residual Refinement Block)
class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_ch + 2*growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_ch + 3*growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_ch + 4*growth, in_ch, 3, 1, 1)
        self.act = nn.SiLU()
        nn.init.zeros_(self.conv5.weight)
        if self.conv5.bias is not None:
            nn.init.zeros_(self.conv5.bias)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        out = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return out

class LRRB(nn.Module):
    def __init__(self, in_ch, mid_ch=64, growth=32, res_scale=0.2):
        super().__init__()
        # Input: concat([z_L (4ch), residual (4ch)]) -> 8 channels
        self.in_conv = nn.Conv2d(in_ch, mid_ch, 1, 1, 0)
        self.db1 = DenseBlock(mid_ch, growth=growth)
        self.db2 = DenseBlock(mid_ch, growth=growth)
        self.db3 = DenseBlock(mid_ch, growth=growth)
        self.out_conv = nn.Conv2d(mid_ch, in_ch // 2, 1, 1, 0)
        self.res_scale = res_scale
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_cat):
        f = self.in_conv(x_cat)
        f = f + self.db1(f) * self.res_scale
        f = f + self.db2(f) * self.res_scale
        f = f + self.db3(f) * self.res_scale
        return self.out_conv(f)


def initialize_unet(rank, return_lora_module_names=False, pretrained_model_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others)

    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
        weight_map=None,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred = noise_pred.to(dtype=torch.float32)
            noise_pred_uncond = noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)
            

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True) + 1e-6

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        diff = (latents - self.stopgrad(latents - grad))**2
        
        if weight_map is not None:
            w_lat = F.interpolate(weight_map, size=latents.shape[-2:], mode='bilinear', align_corners=False)
            loss = (w_lat * diff).mean()
        else:
            loss = diff.mean()

        return loss

    def stopgrad(self, x):
        return x.detach()


class FiDeSR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args

        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others, =\
                    initialize_unet(rank=args.lora_rank_unet, pretrained_model_path=args.pretrained_model_path, return_lora_module_names=True)

            self.lora_rank_unet = args.lora_rank_unet
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others= \
                lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others
            self.lrrb = LRRB(in_ch=args.lrrb_in_ch, mid_ch=args.lrrb_mid_ch, growth=args.lrrb_growth, res_scale=args.lrrb_res_sc).to("cuda")
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = read_yaml(stage1_yaml)
            stage1_args = SimpleNamespace(**stage1_args)
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet = stage1_args.lora_rank_unet
            self.lrrb = LRRB(in_ch=args.lrrb_in_ch, mid_ch=args.lrrb_mid_ch, growth=args.lrrb_growth, res_scale=args.lrrb_res_sc).to("cuda")
            fidesr = torch.load(args.resume_ckpt)
            self.load_ckpt_from_state_dict(fidesr)

        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')

        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

    def set_train(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
            else:
                _p.requires_grad = False
        self.lrrb.train()
        self.lrrb.requires_grad_(True)

    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        self.lora_conf_encoder = LoraConfig(r=sd["lora_rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules"])
        self.lora_conf_decoder = LoraConfig(r=sd["lora_rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules"])
        self.lora_conf_others = LoraConfig(r=sd["lora_rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules"])

        self.unet.add_adapter(self.lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(self.lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(self.lora_conf_others, adapter_name="default_others")

        self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others= \
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])
        # load LRRB weight
        self.lrrb.load_state_dict(sd["lrrb_state_dict"])

    # Adopted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def forward(self, c_t, c_tgt, batch=None, args=None):

        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor

        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        null_prompt_embeds = self.encode_prompt(batch["null_prompt"])

        if random.random() < args.null_text_ratio:
            pos_caption_enc = null_prompt_embeds
        else:
            pos_caption_enc = prompt_embeds

        model_pred = self.unet(
            encoded_control, self.timesteps1,
            encoder_hidden_states=pos_caption_enc.to(torch.float32)
        ).sample
        model_pred = model_pred.to(encoded_control.dtype)

        delta_r = self.lrrb(torch.cat([encoded_control, model_pred], dim=1))
        r_tilde = model_pred + delta_r
        x_denoised = encoded_control - r_tilde

        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds


    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["lora_rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        sd["lrrb_state_dict"] = self.lrrb.state_dict()

        torch.save(sd, outf)


class FiDeSR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
        self.lrrb = LRRB(in_ch=args.lrrb_in_ch, mid_ch=args.lrrb_mid_ch, growth=args.lrrb_growth, res_scale=args.lrrb_res_sc).to(self.device, dtype=self.weight_dtype)
        # Load pretrained weights
        self._load_pretrained_weights(args.pretrained_path)

        # Initialize VAE tiling
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        # Prepare LoRA adapters
        set_weights_and_activate_adapters(self.unet, ["default_encoder", "default_decoder", "default_others"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()

        # Move models to device and precision
        self._move_models_to_device_and_dtype()

        self.timesteps1 = torch.tensor([1], device=self.device).long()

    def _percentile_norm(self, x, lo=0.05, hi=0.95, eps=1e-6):
        # x: [B,1,H,W] or [B,C,H,W]
        B = x.shape[0]
        xf = x.reshape(B, -1).float()  # quantile() requires float32 or float64
        xmin = torch.quantile(xf, lo, dim=1, keepdim=True).view(B,1,1,1).to(x.dtype)
        xmax = torch.quantile(xf, hi, dim=1, keepdim=True).view(B,1,1,1).to(x.dtype)
        return torch.clamp((x - xmin) / (xmax - xmin + eps), 0, 1)

    def _conv3x3_same_rep(self, x, weight):
        # weight: [1,1,3,3], x: [B,1,H,W]
        x_pad = F.pad(x, (1,1,1,1), mode='replicate')
        return F.conv2d(x_pad, weight.to(dtype=x.dtype, device=x.device))

    def _detail_map_from_rgb(self, img_rgb):
        """
        img_rgb: [-1, 1], [B, 3, H, W] or [B, 1, H, W]
        D = mean of enabled components -> percentile normalization -> box blur
        Components:
        - sobel_mag: sqrt(sobel_x^2 + sobel_y^2)
        - laplacian: |conv(lap)|
        - variance : compute local mean with a 3x3 box filter, then apply
                    a 3x3 box filter to (x - mean)^2
        """
        y = (img_rgb * 0.5 + 0.5).to(torch.float32)
        if y.shape[1] == 3:
            gray = 0.299*y[:,0:1] + 0.587*y[:,1:2] + 0.114*y[:,2:3]
        else:
            gray = y

        dev, dt = gray.device, gray.dtype
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=dev, dtype=dt).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=dev, dtype=dt).view(1,1,3,3)
        lap     = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=dev, dtype=dt).view(1,1,3,3)
        box3    = torch.ones((1,1,3,3), device=dev, dtype=dt)/9.0

        comps = []

        ex = self._conv3x3_same_rep(gray, sobel_x)
        ey = self._conv3x3_same_rep(gray, sobel_y)
        sobel_mag = torch.sqrt(ex**2 + ey**2)
        comps.append(sobel_mag)

        lap_mag = self._conv3x3_same_rep(gray, lap).abs()
        comps.append(lap_mag)

        mean = self._conv3x3_same_rep(gray, box3)
        var  = self._conv3x3_same_rep((gray-mean)**2, box3)
        comps.append(var)

        detail_map = sum(comps) / float(len(comps))
        detail_map = self._percentile_norm(detail_map)
        detail_map = self._conv3x3_same_rep(detail_map, box3)
        return detail_map

    def _butterworth_lpf_fft(self, x, rc=0.30, order=2):
        B,C,H,W = x.shape
        X = torch.fft.rfft2(x.to(torch.float32), dim=(-2,-1))
        yy = torch.arange(H, device=x.device).view(H,1) / max(H-1,1)
        xx = torch.arange(W//2+1, device=x.device).view(1,W//2+1) / max(W//2,1)
        r  = torch.sqrt(yy*yy + xx*xx)
        r  = r / r.max().clamp(min=1.0)
        lp = 1.0 / (1.0 + (r/rc).pow(2*order))
        lp = lp.view(1,1,H,W//2+1)
        Y  = X * lp
        y  = torch.fft.irfft2(Y, s=(H,W), dim=(-2,-1))
        return y.to(x.dtype)
    
    def _butterworth_hpf_fft(self, x, rc=0.32, order=2):
        B,C,H,W = x.shape
        X = torch.fft.rfft2(x.to(torch.float32), dim=(-2,-1))
        yy = torch.arange(H, device=x.device).view(H,1)/max(H-1,1)
        xx = torch.arange(W//2+1, device=x.device).view(1,W//2+1)/max(W//2,1)
        r = torch.sqrt(yy*yy + xx*xx)
        r = r / r.max().clamp(min=1.0)
        lp = 1.0 / (1.0 + (r/rc).pow(2*order))
        hp = 1.0 - lp
        hp = hp.view(1,1,H,W//2+1)
        Y = X * hp
        y = torch.fft.irfft2(Y, s=(H,W), dim=(-2,-1))
        return y.to(x.dtype)

    def _make_spatial_gate(self,  c_t, size):
        D = self._detail_map_from_rgb(c_t)
        D_lat = F.interpolate(D, size=size, mode='bilinear', align_corners=False)
        return D_lat.clamp(0, 1)

    def _make_channel_gate(self, encoded_control, rc=0.30, order=2, tau=0.5, sharp=8.0):
        x = encoded_control.float()
        LP = self._butterworth_lpf_fft(x, rc=rc, order=order).float()

        num = LP.pow(2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
        den = x.pow(2).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12).sqrt()

        ratio = (num / den).clamp(0, 1)
        M_ch = torch.sigmoid(float(sharp) * (ratio - float(tau)))

        return M_ch.to(dtype=encoded_control.dtype)

    def _get_dtype(self, precision):
        """Get the appropriate data type based on precision."""
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _move_models_to_device_and_dtype(self):
        """Move models to the correct device and precision."""
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights and initialize LoRA adapters."""
        sd = torch.load(pretrained_path)
        self._load_and_save_ckpt_from_state_dict(sd)

    def _load_and_save_ckpt_from_state_dict(self, sd):
        """Load checkpoint and initialize LoRA adapters."""
        lora_conf_encoder = LoraConfig(r=sd["lora_rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=sd["lora_rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=sd["lora_rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules"])

        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        self.lrrb.load_state_dict(sd["lrrb_state_dict"])

    def set_eval(self):
        """Set models to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    @torch.no_grad()
    def forward(self, c_t, prompt=None):
        """Forward pass for inference."""
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # Tile and process latents if necessary
        model_pred = self._process_latents(encoded_control, prompt_embeds)

        z = encoded_control - model_pred
        zl = z
        zf = z
        lf_active = False
        hf_active = False

        # --- LF enhancement ---
        lf_scale  = getattr(self.args, "lf_scale", 0.1)
        if lf_scale > 0:
            lf_active = True
            lf_rc        = getattr(self.args, "lf_rc", 0.30)
            lf_order     = getattr(self.args, "lf_order", 2)
            lf_tau       = getattr(self.args, "lf_tau", 0.5)
            lf_sharp     = getattr(self.args, "lf_sharp", 8.0)
            lf_dmap_gamma = getattr(self.args, "lf_dmap_gamma", 1.2)

            delta_lp = self._butterworth_lpf_fft(z, rc=lf_rc, order=lf_order)
            M_sp = self._make_spatial_gate(c_t, size=z.shape[-2:])
            
            residual_map = model_pred.abs()
            residual_map = 0.8*residual_map.mean(dim=1, keepdim=True) + 0.2*residual_map.std(dim=1, keepdim=True)
            residual_map = self._percentile_norm(residual_map)

            M_sp_lf = (1.0 - M_sp) * residual_map
            M_sp_lf = self._percentile_norm(M_sp_lf)

            if lf_dmap_gamma != 1.0:
                M_sp_lf = M_sp_lf.pow(lf_dmap_gamma)

            M_ch_lf = self._make_channel_gate(encoded_control, rc=lf_rc, order=lf_order, tau=lf_tau, sharp=lf_sharp)
            M_lf = M_sp_lf * M_ch_lf

            zl = z + lf_scale * M_lf * delta_lp

        # --- HF enhancement ---
        hf_scale   = getattr(self.args, "hf_scale", 0.1)
        if hf_scale > 0:
            hf_active = True
            hf_rc          = getattr(self.args, "hf_rc", 0.30)
            hf_order       = getattr(self.args, "hf_order", 2)
            hf_dmap_gamma  = getattr(self.args, "hf_dmap_gamma", 1.2)

            hpf_z = self._butterworth_hpf_fft(z, rc=hf_rc, order=hf_order)
            delta_hp = hpf_z - self._butterworth_hpf_fft(encoded_control, rc=hf_rc, order=hf_order)

            M_sp_hf = self._make_spatial_gate(c_t, size=z.shape[-2:])
            if hf_dmap_gamma != 1.0:
                M_sp_hf = M_sp_hf.pow(hf_dmap_gamma)

            lf_rc    = getattr(self.args, "lf_rc", 0.30)
            lf_order = getattr(self.args, "lf_order", 2)
            lf_tau   = getattr(self.args, "lf_tau", 0.5)
            lf_sharp = getattr(self.args, "lf_sharp", 8.0)
            M_ch_lf = self._make_channel_gate(encoded_control, rc=lf_rc, order=lf_order, tau=lf_tau, sharp=lf_sharp)
            M_hf = M_sp_hf * (1.0 - M_ch_lf)

            zf = z + hf_scale * M_hf * delta_hp

        if lf_active and hf_active:
            merge_ratio = getattr(self.args, "lf_hf_merge_ratio", 0.5)
            merge_ratio = max(0.0, min(1.0, merge_ratio))
            x_denoised = merge_ratio * zl + (1.0 - merge_ratio) * zf
        elif lf_active:
            x_denoised = zl
        elif hf_active:
            x_denoised = zf
        else:
            x_denoised = z
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        torch.cuda.synchronize()
        total_time = time.time() - start_time

        return total_time, output_image

    def _process_latents(self, encoded_control, prompt_embeds):
        """Process latents with or without tiling."""
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            print("[Tiled Latent]: Input size is small, no tiling required.")
            return self._predict_no_tiling(encoded_control, prompt_embeds)

        print(f"[Tiled Latent]: Input size {h}x{w}, tiling required.")
        return self._predict_with_tiling(encoded_control, prompt_embeds, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds):
        """Predict on the entire latent without tiling."""
        model_pred = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

        delta_r = self.lrrb(torch.cat([encoded_control, model_pred], dim=1))
        model_pred = model_pred + delta_r

        return model_pred

    def _predict_with_tiling(self, encoded_control, prompt_embeds, tile_size, tile_overlap):
        """Predict on the latent with tiling."""
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0
        cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # input tile dimensions
                input_tile = encoded_control[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)
                    # predict the noise residual
                    model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                    model_out = model_out.to(encoded_control.dtype)
                    model_out = model_out + self.lrrb(torch.cat([input_list_t, model_out], dim=1))
                    
                    input_list = []
                noise_preds.append(model_out)

        # Stitch noise predictions for all tiles
        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        model_pred = noise_pred
        return model_pred

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generate a Gaussian mask for tile contributions."""
        from numpy import pi, exp, sqrt
        import numpy as np

        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        """Initialize VAE with tiled encoding/decoding."""
        encoder, decoder = self.vae.encoder, self.vae.decoder

        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward

        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
