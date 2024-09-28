import matplotlib.pyplot as plt
import os
from glob import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_json(json_result: str):
    with open(json_result, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_image(image_path: str):
    return np.array(Image.open(image_path))


if __name__ == '__main__':
    fake_ages_json_path = r'..\diffusion_model\Palette-Image-to-Image-Diffusion-Models-main\fake_ages.json'.replace(
        '\\', '/')
    real_image_path = r'..\dataset\AAF_Face_new\train\real_image'.replace('\\', '/')
    fake_image_path = r'..\dataset\AAF_Face_new\train\fake_image'.replace('\\', '/')
    gather_image_path = r'..\dataset\AAF_Face_new\train\gather_image'.replace('\\', '/')
    os.makedirs(gather_image_path, exist_ok=True)
    # 读取生成的年龄
    fake_ages_dict: dict = load_json(fake_ages_json_path)
    for true_age in tqdm(fake_ages_dict.keys()):
        print(f'==== Age {true_age} ====')
        fake_ages = fake_ages_dict[true_age]
        image_paths = glob(os.path.join(real_image_path, str(true_age), '*'))
        for image_path in image_paths:
            print(image_path)
            fake_images = []
            for fake_age in fake_ages:
                fake_image = os.path.join(fake_image_path, f'{true_age}_{fake_age}', os.path.basename(image_path))
                fake_images.append(fake_image)
            # 绘图
            plt.Figure(figsize=(16,4))
            plt.subplot(1, 4, 1)
            plt.imshow(load_image(image_path))
            plt.title(f'True age: {true_age}')
            plt.subplot(1, 4, 2)
            plt.imshow(load_image(fake_images[0]))
            plt.title(f'Fake age: {fake_ages[0]}')
            plt.subplot(1, 4, 3)
            plt.imshow(load_image(fake_images[1]))
            plt.title(f'Fake age: {fake_ages[1]}')
            plt.subplot(1, 4, 4)
            plt.imshow(load_image(fake_images[2]))
            plt.title(f'Fake age: {fake_ages[2]}')
            plt.suptitle(f'Age {true_age}: {os.path.basename(image_path)}')
            plt.savefig(os.path.join(gather_image_path, f'{true_age}_' + os.path.basename(image_path)))
            # plt.show()
            # break
        # break
