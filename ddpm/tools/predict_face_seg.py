import torch
from models.unet_model.unet_model import UnetModel
from opt.face_options import TrainOptions, TestOptions
from models import create_model
from glob import glob
import os
import numpy as np
from PIL import Image
from data.dataset.aligned_dataset import getImg
import torchvision.transforms as tr
from tqdm import tqdm
from algorithm.fedavg.util import Colorize
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_UNet_for_face(model_path: str):
    opt_train = TrainOptions().parse()
    opt_train.model = 'unet'
    opt_train.federated_algorithm = 'fedavg'
    model = create_model(opt_train)

    state_dict = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(state_dict)
    return model


def predict_by_unet(model: UnetModel, image_base: str, predict_base: str, image_resize_base: str,
                    resize_shape=(384, 384)):
    dir_names = os.listdir(image_base)
    predict_color_base = os.path.join(os.path.dirname(predict_base), os.path.basename(predict_base) + '_color')
    for dir_name in tqdm(dir_names):
        dir_path = os.path.join(image_base, dir_name)
        if not os.path.isdir(dir_path):
            continue
        image_resize_dir_path = os.path.join(image_resize_base, dir_name)
        os.makedirs(image_resize_dir_path, exist_ok=True)
        predict_dir_path = os.path.join(predict_base, dir_name)
        os.makedirs(predict_dir_path, exist_ok=True)
        predict_color_dir_path = os.path.join(predict_color_base, dir_name)
        os.makedirs(predict_color_dir_path, exist_ok=True)

        # 对子文件夹的文件进行推理
        image_paths = glob(os.path.join(dir_path, '*'))

        gray_levels = None
        colors = None

        for image_path in tqdm(image_paths):
            image = getImg(image_path, input_nc=3)
            img = image.resize((resize_shape[0], resize_shape[1]))
            # 保存resize后的图片
            image_resize_path = os.path.join(image_resize_dir_path, os.path.basename(image_path))
            img.save(image_resize_path)
            # 推理
            image = np.array(img)
            img = tr.ToTensor()(image)
            img = img.unsqueeze(0)
            out = model.net(img)
            # 类别数量
            num_classes = out.shape[1]
            # 生成num_classes个灰度级
            if gray_levels is None:
                gray_levels = np.linspace(0, 255, num_classes, dtype=np.uint8)
            # 生成num_classes个随机的RGB颜色元组
            if colors is None:
                colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
            # 取最大值
            out = torch.argmax(out, dim=1)
            # 转numpy
            out = out[0].cpu().numpy()
            # 将类别id转换威威对应灰度级
            out_gray = np.zeros_like(out, dtype=np.uint8)
            for i in range(num_classes):
                out_gray[out == (num_classes - 1 - i)] = gray_levels[num_classes - 1 - i]
            # 将(h,w)形状的out按照id转换为(h,w,3)形状的out
            # out_color = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
            # for i in range(num_classes):
            #     out_color[:, :, 0][out == (num_classes - 1 - i)] = colors[num_classes - 1 - i][0]
            #     out_color[:, :, 1][out == (num_classes - 1 - i)] = colors[num_classes - 1 - i][1]
            #     out_color[:, :, 2][out == (num_classes - 1 - i)] = colors[num_classes - 1 - i][2]

            # 将id与对应颜色的映射关系保存到json文件
            # colors = Colorize(num_classes).cmap[:num_classes]
            # color_maps = {i:colors[i].numpy().tolist()  for i in range(len(colors)) }
            # print(color_maps)
            # with open('/root/data/AI3D/FedST/data_dir/color_maps.json', 'w', encoding='utf-8') as f:
            #     json.dump(color_maps, f, indent=2,ensure_ascii=False)

            label_tensor = Colorize(num_classes)(torch.tensor(out).unsqueeze(0))

            label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
            out_color =  label_numpy.astype(np.uint8)
            # print(out_color.shape)

            # 保存预测结果（灰度图）
            predict_path = os.path.join(predict_dir_path, os.path.basename(image_path))
            Image.fromarray(out_gray.astype(np.uint8)).save(predict_path)
            # 保存预测结果的彩色图
            predict_color_path = os.path.join(predict_color_dir_path, os.path.basename(image_path))
            Image.fromarray(out_color.astype(np.uint8)).save(predict_color_path)

            # print(out.shape)
            # break

        # break


if __name__ == '__main__':
    model_path = '/root/data/AI3D/FedST/dataset/Face_segment/model_fedavg/model49_folds0.pkl'
    image_base = '/root/data/AI3D/FedST/data_dir/selected'
    predict_base = '/root/data/AI3D/FedST/data_dir/selected_predict'
    image_resize_base = '/root/data/AI3D/FedST/data_dir/selected_resize'
    model = load_UNet_for_face(model_path)
    predict_by_unet(model, image_base, predict_base, image_resize_base)
