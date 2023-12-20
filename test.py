import argparse
import os
from PIL import Image
import numpy as np
import torchvision.transforms as tr
from collections import OrderedDict


parser = argparse.ArgumentParser()
# CHAOS_liver   Material   Face_segment
parser.add_argument('--dataset', default='face',
                    choices=['material', 'medical', 'medical_material', 'face', 'aaf_face'],
                    help='Which dataset to use')
parser.add_argument('--model_path', type=str,
                    help='Choose which model to use,unet...')
parser.add_argument('--fold', type=str, default='0', help='chooses fold...')
parser.add_argument('--algorithm', type=str, default='fedddpm',
                    help='Chooses which federated learning algorithm to use')
parser.add_argument('--client_id', type=int, default=-1,
                    help='Client id for local trainer')
parser.add_argument('--bg_value', type=int, default=0,
                    help='Background pixel value for evaluating test set')
parser.add_argument('--save_image', type=int, default=0,
                    help='Whether to save the predicted mask image')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids:e.g.0 0,1,2 1,2 use -1 for CPU')
op = parser.parse_args()
dataset = op.dataset
if dataset == 'material':
    from opt.material_options import TrainOptions, TestOptions
elif dataset == 'face':
    from opt.face_options import TrainOptions, TestOptions
elif dataset == 'medical':
    from opt.medical_options import TrainOptions, TestOptions
elif op.dataset == 'medical_material':
    from opt.medical_material_options import TrainOptions, TestOptions
elif op.dataset == 'aaf_face':
    from opt.aaf_face_options import TrainOptions, TestOptions

fold = op.fold
fed_methods = op.algorithm
model_path = op.model_path + fold + '.pkl'
method = dataset + '_' + fed_methods + '_fold_' + fold
client_id = op.client_id
bg_value = op.bg_value
from data.dataloader.data_loader import CreateDataLoader
from models import create_model
import time
from util.util import mkdir
from util.metrics import get_total_evaluation
import torch
import tqdm

is_save_image = True if op.save_image>0 else False

if __name__ == '__main__':
    gpu_ids = op.gpu_ids.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_ids])
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    opt = TestOptions().parse()
    opt_train = TrainOptions().parse()

    # hard-code some parameters for test
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.client_id = client_id

    if opt.no_normalize:
        transform = tr.ToTensor()
    else:
        transform = tr.Compose([tr.ToTensor(),
                                tr.Normalize(mean=0.5,
                                             std=0.5)
                                ])
    data_loader = CreateDataLoader(opt, dataroot=opt.dataroot, image_dir=opt.test_img_dir, \
                                   label_dir=opt.test_label_dir, record_txt=None, transform=transform, is_aug=False)
    print(opt.dataroot)
    print(opt.test_img_dir)
    print(opt.test_label_dir)

    dataset = data_loader.load_data()
    datasize = len(data_loader)
    print('#test images = %d, batchsize = %d' % (datasize, opt.batch_size))
    state_dict = torch.load(model_path, map_location='cuda:0')
    sd = OrderedDict()

    # opt.isTrain = True
    opt_train.model = 'unet'
    model = create_model(opt_train)

    # if 'fedst' in method:
    #     for i in state_dict.keys():
    #         if 'netDseg' in i:
    #             key = '.'.join(['net'] + i.split('.')[1:])
    #             sd[key] = state_dict[i]
    #     model.load_state_dict(sd)
    # else:
    #     model.load_state_dict(state_dict)
    model.load_state_dict(state_dict)

    print(f"[Test] Model {model_path} has loaded.")

    # img_dir = os.path.join(opt.results_dir, opt.name, '%s' % method)
    save_dir_name = '_'.join(str(op.model_path).split('/')[-2:]) + '_' + fold
    img_dir = os.path.join(opt.results_dir, opt.name, save_dir_name)
    print('Eval result has saved to %s' % img_dir)
    mkdir(img_dir)

    eval_results = {}
    count = 0
    with open(img_dir + '_eval.txt', 'w') as log:
        now = time.strftime('%c')
        log.write('=============Evaluation (%s)=============\n' % now)

    result = []

    for i, data in enumerate(tqdm.tqdm(dataset)):
        c_idx = 0

        model.set_input(data)
        with torch.no_grad():
            pred = model()[0]

        pred = pred.max(0)[1].cpu().numpy()
        img_path = data['path'][0]
        short_path = os.path.basename(img_path)
        name = os.path.splitext(short_path)[0]
        image_name = '%s.png' % name
        if is_save_image:
            mask_dir = os.path.join(img_dir, 'mask')
            os.makedirs(mask_dir, exist_ok=True)
            Image.fromarray(pred.astype(np.uint8)).save(os.path.join(mask_dir, image_name))

        eval_start = time.time()
        count += 1
        mask = data['label'].squeeze().numpy().astype(np.uint8)
        print(f"bg_value: {bg_value}")
        eval_result = get_total_evaluation(pred, mask, bg_value=bg_value)
        print(data['label'].data[0].shape)

        message = '%04d: %s \t' % (count, name)
        for k, v in eval_result.items():
            if k in eval_results:
                eval_results[k] += v
            else:
                eval_results[k] = v
            message += '%s: %.5f\t' % (k, v)

        eval_result['name'] = name
        result.append(eval_result)
        print(message, 'cost: %.4f' % (time.time() - eval_start))
        with open(img_dir + '_eval.txt', 'a') as log:
            log.write(message + '\n')

    message = 'total %d:\n' % count
    for k, v in eval_results.items():
        message += 'm_%s: %.5f\t' % (k, v / count)
    print(message)
    with open(img_dir + '_eval.txt', 'a') as log:
        log.write(message + '\n')
