from PIL import Image
import numpy as np
import os
from util.metrics import get_total_evaluation
import torch
from models import create_model
from opt.aaf_face_new_options import TrainOptions, TestOptions
from collections import OrderedDict
from torchvision import transforms as tr
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

transform = tr.ToTensor()


def load_model(model_path: str, method='feddpm'):
    opt_train = TrainOptions().parse()
    opt_train.model = 'unet'
    model = create_model(opt_train)
    # 加载参数
    sd = OrderedDict()
    state_dict = torch.load(model_path, map_location='cuda:3')
    if 'fedst' in method:
        for i in state_dict.keys():
            if 'netDseg' in i:
                key = '.'.join(['net'] + i.split('.')[1:])
                # key = i
                sd[key] = state_dict[i]
        model.load_state_dict(sd)
    else:
        model.load_state_dict(state_dict)
    return model


def load_data(image_path: str, with_label=True):
    image = transform(np.array(Image.open(image_path)))
    label = None
    if with_label:
        label = transform(np.array(Image.open(image_path.replace('real_image', 'real_label'))))
    return image, label


def get_total_eval(model, real_image_dir: str, real_label_dir: str):
    for client_id in tqdm(range(79)):
        image_paths = glob(os.path.join(real_image_dir, str(client_id), '*'))
        for image_path in tqdm(image_paths):
            image, label = load_data(image_path, with_label=True)
            image = torch.unsqueeze(image, dim=0)
            input_data = {}
            input_data['image'] = image
            input_data['label'] = label
            input_data['path'] = None
            model.set_input(input_data)
            pred = model()[0]
            pred = pred.max(0)[1].cpu().numpy()
            mask = (label * 255).squeeze().numpy().astype(np.uint8)
            # print(np.unique(mask))
            # print(pred.shape)
            eval_result = get_total_evaluation(pred, mask)
            # print(eval_result)
            mpa = eval_result['mpa']
            dice = eval_result['dice']
            total_result[image_path] = {'mpa': mpa, 'dice': dice}
        save_json(json_result, total_result)


def save_json(json_result: str, data):
    with open(json_result, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(json_result:str):
    with open(json_result, 'r' ,encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_top_path(json_result:str, k=10):
    data = load_json(json_result)
    top_k = sorted(data.items(), key=lambda x: x[1]['mpa'], reverse=True)[:k]
    print('mpa:',top_k)
    for item in top_k:
        print(item)
    top_k = sorted(data.items(), key=lambda x: x[1]['dice'], reverse=True)[:k]
    print('dice:',top_k)
    for item in top_k:
        print(item)



if __name__ == '__main__':
    data_mode = 'test'
    model_path = '../dataset/AAF_Face_new/model_fedddpm_0811_074201/model49_folds0.pkl'
    real_image_dir = f'../dataset/AAF_Face_new/{data_mode}/real_image'
    real_label_dir = f'../dataset/AAF_Face_new/{data_mode}/real_label'
    json_result = f'./predict_{data_mode}.json'
    total_result = {}  # 所有推理结果
    client_id = 0
    # model = load_model(model_path)
    # model.eval()
    # get_total_eval(model, real_image_dir, real_label_dir)
    get_top_path(json_result,k=30)
