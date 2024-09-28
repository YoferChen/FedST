import shutil
from glob import glob
import os
from tqdm import tqdm

select_image_dir = r'..\dataset\AAF_Face_new\train\gather_image_select'
real_image_dir = r'..\dataset\AAF_Face_new\train\real_image'
real_label_dir = r'..\dataset\AAF_Face_new\train\real_color'
fake_image_dir = r'..\dataset\AAF_Face_new\train\fake_image'
target_dir = r'..\dataset\AAF_Face_new\train\style_transfer_select'


def image_collection():
    image_paths = glob(os.path.join(select_image_dir, '*'))
    # print(image_paths)
    for image_path in tqdm(image_paths):
        client_id, filename = os.path.basename(image_path).split('_')
        # name,ext = os.path.splitext(filename)
        target_collection_dir = os.path.join(target_dir, os.path.splitext(filename)[0])
        os.makedirs(target_collection_dir, exist_ok=True)
        print(filename)
        # 原图
        real_image_path = os.path.join(real_image_dir, client_id, filename)
        shutil.copy(real_image_path, os.path.join(target_collection_dir, filename))
        # 标签
        label_image_path = os.path.join(real_label_dir, client_id, filename)
        shutil.copy(label_image_path, os.path.join(target_collection_dir, 'label_' + filename))
        # fake图像
        dir_names = os.listdir(fake_image_dir)
        for dirname in dir_names:
            true_age, fake_age = dirname.split('_')
            if true_age == client_id:
                fake_image_path = os.path.join(fake_image_dir, dirname, filename)
                shutil.copy(fake_image_path, os.path.join(target_collection_dir, f'style{fake_age}_' + filename))


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

save_dir = r'..\dataset\AAF_Face_new\train\style_transfer_select_allInOne'
os.makedirs(save_dir, exist_ok=True)


def show_result():
    dir_paths = glob(os.path.join(target_dir, '*'))
    for dir_path in tqdm(dir_paths):
        image_paths = glob(os.path.join(dir_path, '*'))
        plt.figure(figsize=(20, 4))
        for idx, image_path in enumerate(image_paths):
            # print(image_path)
            title = os.path.basename(image_path).split('_')[0]
            plt.subplot(151 + idx)
            plt.imshow(np.array(Image.open(image_path)))
            plt.title(title)
        plt.suptitle(os.path.basename(dir_path))
        plt.savefig(os.path.join(save_dir, os.path.basename(dir_path) + '.png'))
        # plt.show()
        # break


if __name__ == '__main__':
    image_collection()
    show_result()
