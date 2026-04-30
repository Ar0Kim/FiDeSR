#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import glob

from fidesr import FiDeSR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix


def fidesr_test(args):
    # Initialize the model
    model = FiDeSR_eval(args)
    model.set_eval()

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            image_names.extend(sorted(glob.glob(os.path.join(args.input_image, ext))))
            image_names.extend(sorted(glob.glob(os.path.join(args.input_image, ext.upper()))))
    else:
        image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # Empty prompt (no RAM)
        validation_prompt = ''

        # Translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
            inference_time, output_image = model(c_t, prompt=validation_prompt)

        print(f"Inference time: {inference_time:.4f} seconds")
        time_records.append(inference_time)

        output_image = output_image * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=output_pil, source=input_image)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=output_pil, source=input_image)

        if resize_flag:
            output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
        output_pil.save(os.path.join(args.output_dir, bname))

    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, required=True, help="path to the input image or directory")
    parser.add_argument('--output_dir', '-o', type=str, required=True, help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, required=True, help="path to a LoRA checkpoint .pkl")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp16")

    parser.add_argument("--lf_scale", type=float, default=0.2, help="LF strength")
    parser.add_argument("--lf_rc", type=float, default=0.10, help="LF Butterworth LPF cutoff ratio in [0,1]")
    parser.add_argument("--lf_order", type=int, default=2, help="LF Butterworth order")
    parser.add_argument("--lf_tau", type=float, default=0.8, help="Channel gate threshold")
    parser.add_argument("--lf_sharp", type=float, default=10.0, help="Channel gate steepness")
    parser.add_argument("--lf_dmap_gamma", type=float, default=1.2,
                        help="Exponent for LF spatial D-map gating (>=1 emphasizes edges)")

    parser.add_argument("--hf_scale", type=float, default=0.3, help="HF strength")
    parser.add_argument("--hf_rc", type=float, default=0.32, help="HF Butterworth HPF cutoff ratio [0,1]")
    parser.add_argument("--hf_order", type=int, default=2, help="HF Butterworth order")
    parser.add_argument("--hf_dmap_gamma", type=float, default=1.2,
                        help="Exponent for HF spatial D-map gating (>=1 emphasizes edges)")

    parser.add_argument("--lrrb_in_ch", type=int, default=8)
    parser.add_argument("--lrrb_mid_ch", type=int, default=64)
    parser.add_argument("--lrrb_growth", type=int, default=32)
    parser.add_argument("--lrrb_res_sc", type=float, default=0.2)

    args = parser.parse_args()

    # Call the processing function
    fidesr_test(args) 