from glob import glob
import os
import random
import shutil
from tqdm import tqdm
import json


def select_file_each_class(data_dir: str, num_each_class: int):
    '''
    data_dir文件夹中的文件名以XXXA类别名.jpg命名的，需要先划分各个类别的文件路径到一个dict，然后从各个类别的文件总随机筛选按num_each_class个文件，返回一个dict，包含各个类别选择的文件路径
    '''
    file_list = glob(os.path.join(data_dir, '*'))
    # print(file_list)
    class_dict = {}
    for file_path in file_list:
        class_name = file_path.split('A')[-1].split('.')[0]
        if class_dict.get(class_name):
            class_dict[class_name].append(file_path)
        else:
            class_dict[class_name] = [file_path]

    # 不筛选
    if num_each_class is None:
        return class_dict

    select_dict = {}
    for class_name in class_dict.keys():
        select_dict[class_name] = random.sample(class_dict[class_name], num_each_class)

    return select_dict


def save_select_json(select_dict: dict, save_path: str):
    '''
    将上一个函数筛选的文件路径保存到json文件中
    '''
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(select_dict, f, indent=2, ensure_ascii=False)


def copy_file(select_dict: dict, save_dir: str):
    '''
    将上一个函数筛选的文件复制到save_dir文件夹中，按类别创建子文件夹
    '''
    for class_name in tqdm(select_dict.keys()):
        class_dir = os.path.join(save_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for file_path in select_dict[class_name]:
            shutil.copy(file_path, class_dir)


if __name__ == '__main__':
    # data_dir = '../data_dir/original images'
    # save_dir = '../data_dir/selected'
    # os.makedirs(save_dir, exist_ok=True)
    # json_path = os.path.join(save_dir, 'select.json')
    # select_files = select_file_each_class(data_dir, 20)
    # save_select_json(select_files, json_path)
    # copy_file(select_files, save_dir)

    # data_dir = '../data_dir/selected_resize_total_predict_mask'
    # save_dir = '../data_dir/selected_resize_predict_mask'
    data_dir = '../data_dir/selected_resize_total_json'
    save_dir = '../data_dir/selected_resize_json'
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, 'select.json')
    select_files = select_file_each_class(data_dir, None)
    # save_select_json(select_files, json_path)
    copy_file(select_files, save_dir)
