# -*- coding: utf-8 -*-
# 导入工具包
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import shutil
from glob import glob
from tqdm import tqdm


# import torch
# from PIL import Image
#
# from torchvision.models.feature_extraction import create_feature_extractor
# # 有 GPU 就用 GPU，没有就用 CPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('device', device)
#
# from torchvision import transforms
# # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
# test_transform = transforms.Compose([transforms.Resize(256),
#                                      transforms.CenterCrop(224),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(
#                                          mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#                                     ])
#
# model = torch.load('face_pytorch_20221107.pth', map_location=torch.device('cpu'))
# model = model.eval().to(device)
# model_trunc = create_feature_extractor(model, return_nodes={'avgpool': 'semantic_feature'})
def process_aaf_face_new(client_ids: list = [8, 38, 68], with_fake=True):
    data_dir = '../dataset/AAF_Face_new/train'
    target_dir = './AAF_face_new'
    if with_fake:
        target_dir = os.path.join(target_dir, 'real_image_with_fake')
    else:
        target_dir = os.path.join(target_dir, 'real_image')
    real_image_dir = os.path.join(data_dir, 'real_image')
    fake_image_dir = os.path.join(data_dir, 'fake_image')

    for client_id in tqdm(client_ids):
        targe_client_dir = os.path.join(target_dir, str(client_id))
        os.makedirs(targe_client_dir, exist_ok=True)
        # 复制原图像
        client_image_dir = os.path.join(real_image_dir, str(client_id))
        image_paths = glob(os.path.join(client_image_dir, '*'))
        for image_path in image_paths:
            shutil.copy(image_path, os.path.join(targe_client_dir, os.path.basename(image_path)))
        # 复制fake图像（增加文件名后缀_style_$age）
        if with_fake:
            fake_dirs = glob(os.path.join(fake_image_dir, f'{client_id}_*'))
            for fake_dir in fake_dirs:
                fake_images = glob(os.path.join(fake_dir, '*'))
                style_name = os.path.basename(fake_dir).split('_')[1]
                for fake_image in fake_images:
                    name, ext = os.path.splitext(os.path.basename(fake_image))
                    shutil.copy(fake_image, os.path.join(targe_client_dir, name + f'_style{style_name}' + ext))


# process_aaf_face_new(list(range(79)), with_fake=True)
# exit(0)

experiment_index = 3  # 0/1/2
# dataset_name = ["材料", "人脸", "医学"]

# aaf_dir_list = ['0', '1', '2']
aaf_dir_list = [str(i) for i in range(79)]
# aaf_class_list = ['Age 02-11', 'Age 32-41',  'Age 62-71']
aaf_class_list = [str(i) for i in range(79)]

dataset_name = ["material", "face", "medical", "aaf_face", "aaf_face", "aaf_face", "aaf_face"]
path_list = ["material/real_image", "face-style-transform/train/real_image", "CHAOS-style-transform/train/real_image",
             "AAF_face_new/real_image_gather", "AAF_face_new/real_image_with_fake_gather",
             "AAF_face_new/real_image", "AAF_face_new/real_image_with_fake"]
all_image_dir_list = [["0", "1"], ["0", "1", "2"], ["0", "1"], aaf_dir_list, aaf_dir_list, aaf_dir_list, aaf_dir_list]
all_class_list = [["Austenite", "IronCrystal"], ["African face", "American face", "Asian face"], ["CT", "MRI"],
                  aaf_class_list, aaf_class_list, aaf_class_list, aaf_class_list]
dataset = dataset_name[experiment_index]
image_dir_path = path_list[experiment_index]
image_dir_list = all_image_dir_list[experiment_index]
class_list = all_class_list[experiment_index]
n_class = len(class_list)  # 图像标签类别数


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# 载入图像特征
image_array = []
for i in tqdm(image_dir_list):
    image_path = os.path.join(image_dir_path, i)
    image_list = os.listdir(image_path)
    if ".ipynb_checkpoints" in image_list:
        image_list.remove(".ipynb_checkpoints")
    for image in tqdm(image_list):
        image_data_path = os.path.join(image_path, image)
        image_data = cv2.imread(image_data_path, cv2.IMREAD_COLOR)
        image_data = np.resize(image_data, (384, 384, 3))
        # print(image_data.shape)
        # image_data = Image.fromarray(image_data)
        # input_img = test_transform(image_data)  # 预处理
        # input_img = input_img.unsqueeze(0).to(device)
        # # 执行前向预测，得到指定中间层的输出
        # pred_logits = model_trunc(input_img)
        # pred_logits = pred_logits['semantic_feature'].squeeze().detach().cpu().numpy()
        # # print(image_data.shape)
        # pred_logits = pred_logits.reshape(-1)
        image_data = image_data.reshape(-1)  # 铺平为1维
        # print(image_data.shape)
        image_data = image_data.astype(np.float32)
        image_data = normalization(image_data)
        # print(image_data.shape)
        # image_data = list(image_data)
        image_array.append(image_data)

image_array = np.array(image_array)
print(image_array.shape)  # (980, 442368)

# 可视化配置
import seaborn as sns

marker = '.'

palette = np.array(sns.color_palette("bright", n_class))  # sns.hls_palette(n_class)  # 配色方案
sns.palplot(palette)
# t-SNE降维
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, n_iter=1000, init="pca", random_state=0)
X_tsne_2d = tsne.fit_transform(image_array)  # 需要大内存。否则报错：MemoryError: bad allocation

print(X_tsne_2d.shape)

# 可视化展示
plt.figure(figsize=(14, 14))
for idx, each_type in enumerate(class_list):  # 遍历每个类别
    # 获取颜色和点型
    color = palette[idx]
    # 判断各类别图像的数量
    image_num = int(X_tsne_2d.shape[0] / n_class)
    # 绘制散点图
    plt.scatter(X_tsne_2d[idx * image_num:(idx + 1) * image_num, 0],
                X_tsne_2d[idx * image_num:(idx + 1) * image_num, 1], color=color, marker=marker, label=each_type, s=150)
    # if idx == 0:
    #     plt.scatter(X_tsne_2d[0:1400, 0],
    #                 X_tsne_2d[0:1400, 1], color=color, marker=marker, label=each_type, s=150)
    # else:
    #     plt.scatter(X_tsne_2d[1400:, 0],
    #                 X_tsne_2d[1400:, 1], color=color, marker=marker, label=each_type, s=150)
plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1), loc='upper right')
plt.xticks([])
plt.yticks([])
# plt.savefig(dataset + '风格迁移数据集t-SNE二维降维可视化.pdf', dpi=300)  # 保存图像
plt.savefig(dataset + f'style_transfer_t-SNE_origin_class{len(all_class_list[experiment_index])}.pdf', dpi=300)  # 保存图像
plt.savefig(dataset + f'style_transfer_t-SNE_origin_class{len(all_class_list[experiment_index])}.png', dpi=300)  # 保存图像
# plt.savefig(dataset + f'style_transfer_t-SNE_with_style_transfer_class{len(all_class_list)}.pdf', dpi=300)  # 保存图像
# plt.savefig(dataset + f'style_transfer_t-SNE_with_style_transfer_class{len(all_class_list)}.png', dpi=300)  # 保存图像
plt.show()
