import json
import os
from collections import OrderedDict
import copy


def create_configs():
    config_dir = '/chenyongfeng/FedST/diffusion_model/Palette-Image-to-Image-Diffusion-Models-main/config'
    AAF_separate_config_dir = os.path.join(config_dir, 'AAFnew_separate')
    os.makedirs(AAF_separate_config_dir, exist_ok=True)

    base_config = os.path.join(config_dir, 'AAFnew0_train.json')

    json_str = ''
    with open(base_config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt_base = json.loads(json_str, object_pairs_hook=OrderedDict)

    for i in range(79):
        opt = copy.deepcopy(opt_base)
        opt['name'] = f"labeltoimage_AAFnew{i}"
        opt['datasets']['train']['which_dataset']['args']['data_flist'] = f"datasets/AAF_Face_new/flist/train.flist{i}"
        opt['datasets']['train']['which_dataset']['args']['data_client'] = f"{i}"
        opt['train']['n_epoch'] = 300
        opt['path']['resume_state'] = "experiments/train_CelebaHQ_label2image_230809_135721/checkpoint/40"

        save_path = os.path.join(AAF_separate_config_dir, f'labeltoimage_AAFnew{i}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(opt, f, ensure_ascii=False, indent=4)


def create_sh(task_num=4):
    task_num_for_each_gpu = 80 // task_num
    total_num = 0
    for i in range(task_num):
        base_dir = '/chenyongfeng/FedST/diffusion_model/Palette-Image-to-Image-Diffusion-Models-main'
        sh_path = os.path.join(base_dir, f'train_AAFnew{i + 1}.sh')
        with open(sh_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n\n")
            f.write('echo "Start running train..."\n')
            for t in range(total_num, total_num + task_num_for_each_gpu):
                f.write(
                    f"python ./run.py -c config/AAFnew_separate/labeltoimage_AAFnew{t}.json -p train --gpu_ids {i}\n")
            f.write('echo "Scripts have finished running."')
        total_num += task_num_for_each_gpu


if __name__ == '__main__':
    create_configs()
    create_sh()
