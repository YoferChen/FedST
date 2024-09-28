from glob import glob
import os
import json
from tqdm import tqdm
import shutil

diffusion_code_dir = '/chenyongfeng/FedST/diffusion_model/Palette-Image-to-Image-Diffusion-Models-main-join'
result_base_dir = os.path.join(diffusion_code_dir, 'experiments/test_result_join_0812')
json_path = os.path.join(diffusion_code_dir, 'fake_ages.json')
target_dataset_dir = '/chenyongfeng/FedST/dataset/AAF_Face_new/train/fake_image_join'


def copy_fake_image():
    fake_ages = json.load(open(json_path, 'r', encoding='utf-8'))
    for true_age in tqdm(fake_ages.keys()):
        print('True age:', true_age)
        for fake_age in fake_ages[true_age]:
            fake_age_dir = glob(os.path.join(result_base_dir, f'test_AAFnew_data{true_age}_style{fake_age}*'))[0]
            fake_age_dir = os.path.join(fake_age_dir, 'results', 'test', '0')
            target_dir = os.path.join(target_dataset_dir, f'{true_age}_{fake_age}')
            print(f'Copy {fake_age_dir} to {target_dir}')
            os.makedirs(target_dir,exist_ok=True)
            src_images = glob(os.path.join(fake_age_dir, '*.png'))
            for src_image in src_images:
                if '_' in os.path.basename(src_image):
                    continue
                target_path = os.path.join(target_dir, os.path.basename(src_image))
                shutil.copy(src_image, target_path)


if __name__ == '__main__':
    copy_fake_image()
