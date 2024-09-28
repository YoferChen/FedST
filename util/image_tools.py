import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
def image2binary(image_dir:str, save_dir:str):
    """
    将图片转换为二值图
    :param image_dir: 图片文件夹
    :param save_dir: 保存文件夹
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_list = os.listdir(image_dir)
    for image_name in tqdm(image_list):
        img = Image.open(os.path.join(image_dir, image_name))
        image_array = np.array(img, dtype=bool)
        img = Image.fromarray(image_array)
        img.save(os.path.join(save_dir, image_name))

if __name__ == '__main__':
    # image_dir = '/root/data/AI3D/FedST_Public/results/XCX/model_fedprox_0131_174324_model50_folds_0/mask'
    # save_dir = '/root/data/AI3D/FedST_Public/results/XCX/model_fedprox_0131_174324_model50_folds_0/mask_binary'
    image_dir = '/root/data/AI3D/FedST_Public/results/XCX/model_fedprox_0131_174324_model99_folds_0/mask'
    save_dir = '/root/data/AI3D/FedST_Public/results/XCX/model_fedprox_0131_174324_model99_folds_0/mask_binary'
    image2binary(image_dir, save_dir)
