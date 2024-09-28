import os
from glob import glob
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    # image_base_dir = '/chenyongfeng/FedST/diffusion_model/Palette-Image-to-Image-Diffusion-Models-main/datasets/AAF_Face/test/real_image'
    image_base_dir = r'D:\Projects\AI3D\cvpr_code\diffusion_model\Palette-Image-to-Image-Diffusion-Models-main\datasets\AAF_Face\test\real_image'
    sub_dirs = os.listdir(image_base_dir)
    for sub_dir in tqdm(sub_dirs):
        sub_dir_path = os.path.join(image_base_dir, sub_dir)
        image_paths = glob(os.path.join(sub_dir_path, '*.jpg'))
        for image_path in image_paths:
            Image.open(image_path).save(os.path.splitext(image_path)[0] + '.png')
            os.remove(image_path)
