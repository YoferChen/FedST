from glob import glob
import os
import shutil
from tqdm import tqdm
import random
from PIL import Image

if __name__ == '__main__':
    image_dir = r'C:\Users\17325\wisdom_store_workspace\Local\FedST_Face_align\data\raw_data'
    mask_dir = r'C:\Users\17325\wisdom_store_workspace\Local\FedST_Face_align\data\labeled_mask'
    target_dir = r'D:\Projects\AI3D\cvpr_code\dataset\AAF_Face_resplit'
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')

    image_paths = glob(os.path.join(image_dir, '*'))
    ratio = 0.8
    data_dict = {}
    # 各个类别的数据归类统计
    for image_path in image_paths:
        class_name = os.path.splitext(os.path.basename(image_path))[0].split('A')[-1]
        if data_dict.get(class_name):
            data_dict[class_name].append(image_path)
        else:
            data_dict[class_name] = [image_path]
    print(data_dict)
    for key in data_dict:
        print(f'{key}: {len(data_dict[key])}')
    # 按照比例划分训练集和测试集
    for class_name in tqdm(data_dict):
        class_id = int(class_name) - 2  # id=年龄-2
        images = data_dict[class_name]
        train_num = int(len(images) * ratio)
        test_num = len(images) - train_num
        # 图像列表打乱
        random.shuffle(images)
        train_images = images[:train_num]
        test_images = images[train_num:]
        # 划分数据集
        for images, output_dir in zip([train_images, test_images], [train_dir, test_dir]):
            # 创建训练集和测试集文件夹
            real_image_dir = os.path.join(output_dir, 'real_image', f'{class_id}')
            real_label_dir = os.path.join(output_dir, 'real_label', f'{class_id}')
            os.makedirs(real_image_dir, exist_ok=True)
            os.makedirs(real_label_dir, exist_ok=True)
            for image in images:
                # shutil.copy(image, real_image_dir)
                Image.open(image).save(os.path.join(real_image_dir, os.path.basename(image).split('.')[0] + '.png'))
                # 匹配mask
                mask_path = glob(os.path.join(mask_dir, os.path.splitext(os.path.basename(image))[0] + '.*'))[0]
                shutil.copy(mask_path, real_label_dir)
