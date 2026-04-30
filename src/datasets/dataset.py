import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from pathlib import Path

import numpy as np
from src.datasets.realesrgan import RealESRGAN_degradation



class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
                transforms.RandomHorizontalFlip(),
            ])
            with open(args.dataset_txt_paths, 'r') as f:
                self.gt_list = [line.strip() for line in f.readlines()]
            if args.highquality_dataset_txt_paths is not None:
                with open(args.highquality_dataset_txt_paths, 'r') as f:
                    self.hq_gt_list = [line.strip() for line in f.readlines()]

        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.lr_list = []
            self.gt_list = []
            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))
            self.crop_preproc = transforms.Compose([
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            if self.args.highquality_dataset_txt_paths is not None:
                if np.random.uniform() < self.args.prob:
                    gt_path = self.gt_list[idx]
                else:
                    rand_idx = random.choice(range(len(self.hq_gt_list)))
                    gt_path = self.hq_gt_list[rand_idx]
            else:
                gt_path = self.gt_list[idx]
                
            gt_img = Image.open(gt_path).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {
                "neg_prompt": self.args.neg_prompt_csd,
                "null_prompt": "",
                "output_pixel_values": output_t,
                "conditioning_pixel_values": img_t,
            }
            return example
                
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t = self.crop_preproc(input_img)
            output_t = self.crop_preproc(output_img)
                
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {
                "neg_prompt": self.args.neg_prompt_csd,
                "null_prompt": "",
                "output_pixel_values": output_t,
                "conditioning_pixel_values": img_t,
                "base_name": os.path.basename(self.lr_list[idx]),
            }
            return example