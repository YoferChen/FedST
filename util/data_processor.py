'''
实验数据处理工具集合
2024.02.27
浅若清风cyf
'''

import shutil
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
'''
从合成图像文件夹提取合成图像复制到指定文件夹
'''

def extract_fake_images(src_dir:str, target_dir:str, match_rule):
    '''
    从合成图像文件夹提取合成图像复制到指定文件夹
    :param src_dir: 合成图像文件夹
    :param target_dir: 目标文件夹
    :param match_rule: 匹配规则函数，传入图像文件名，返回True或False
    :return:
    '''
    os.makedirs(target_dir, exist_ok=True)
    files = glob(os.path.join(src_dir, '*'))
    for file in tqdm(files):
        name = os.path.basename(file)
        if match_rule(name):
            shutil.copy(file, os.path.join(target_dir, name))

def match_rule(name:str):
    '''
    匹配规则函数，传入图像文件名，返回True或False
    :param name: 图像文件名
    :return: True或False
    '''
    # if 'GT_' in name or 'Process_' in name:
    #     return False
    return True

if __name__ == '__main__':
    data_pairs = [
        {
            'src': r'D:\Projects\AI3D\cvpr_code\diffusion_model\Palette-Image-to-Image-Diffusion-Models-main\experiments\test_labeltoimage_material_average_data0_240221_125017\fake_images',
            'target': r'D:\Projects\AI3D\FedST\FedST_Public\dataset\Material\train\fake_image_average\0',
        },
        {
            'src': r'D:\Projects\AI3D\cvpr_code\diffusion_model\Palette-Image-to-Image-Diffusion-Models-main\experiments\test_labeltoimage_material_average_data1_240221_124036\fake_images',
            'target': r'D:\Projects\AI3D\FedST\FedST_Public\dataset\Material\train\fake_image_average\1',
        },
    ]
    for data_pair in tqdm(data_pairs):
        extract_fake_images(data_pair['src'], data_pair['target'], match_rule)

