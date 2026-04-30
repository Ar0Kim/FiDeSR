import os

def write_png_paths(folder_paths, txt_path):
    with open(txt_path, 'w') as f:
        for folder_path in folder_paths:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.png'):
                        full_path = os.path.join(root, file)
                        f.write(full_path + '\n')

folder_paths = [
    "preset/train_dataset/train/DIV2K/DIV2K_train_HR",
    "preset/train_dataset/train/FFFQ_10000",
    "preset/train_dataset/train/Flickr2k/Flickr2K",
    "preset/train_dataset/train/LSDIR",
]

txt_path = "preset/gt_path.txt"

write_png_paths(folder_paths, txt_path)
