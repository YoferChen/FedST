import json
import os
from collections import OrderedDict
import copy
from glob import glob
from tqdm import tqdm
import numpy as np
import json
base_dir = '/chenyongfeng/FedST/diffusion_model/Palette-Image-to-Image-Diffusion-Models-main-join'

model_dict = {0: {'name': 'train_labeltoimage_AAFnew0_230809_163118', 'epoch': 300},
              1: {'name': 'train_labeltoimage_AAFnew1_230809_165808', 'epoch': 300},
              2: {'name': 'train_labeltoimage_AAFnew2_230809_172458', 'epoch': 150},
              3: {'name': 'train_labeltoimage_AAFnew3_230809_175152', 'epoch': 250},
              4: {'name': 'train_labeltoimage_AAFnew4_230809_181859', 'epoch': 300},
              5: {'name': 'train_labeltoimage_AAFnew5_230809_184539', 'epoch': 100},
              6: {'name': 'train_labeltoimage_AAFnew6_230809_191237', 'epoch': 300},
              7: {'name': 'train_labeltoimage_AAFnew7_230809_193928', 'epoch': 250},
              8: {'name': 'train_labeltoimage_AAFnew8_230809_200616', 'epoch': 200},
              9: {'name': 'train_labeltoimage_AAFnew9_230809_203304', 'epoch': 150},
              10: {'name': 'train_labeltoimage_AAFnew10_230809_205948', 'epoch': 100},
              11: {'name': 'train_labeltoimage_AAFnew11_230809_212638', 'epoch': 250},
              12: {'name': 'train_labeltoimage_AAFnew12_230809_215317', 'epoch': 300},
              13: {'name': 'train_labeltoimage_AAFnew13_230809_221956', 'epoch': 300},
              14: {'name': 'train_labeltoimage_AAFnew14_230809_224650', 'epoch': 300},
              15: {'name': 'train_labeltoimage_AAFnew15_230809_231333', 'epoch': 300},
              16: {'name': 'train_labeltoimage_AAFnew16_230809_234022', 'epoch': 200},
              17: {'name': 'train_labeltoimage_AAFnew17_230810_000703', 'epoch': 150},
              18: {'name': 'train_labeltoimage_AAFnew18_230810_003355', 'epoch': 300},
              19: {'name': 'train_labeltoimage_AAFnew19_230810_010047', 'epoch': 300},
              20: {'name': 'train_labeltoimage_AAFnew20_230809_163221', 'epoch': 50},
              21: {'name': 'train_labeltoimage_AAFnew21_230809_165903', 'epoch': 200},
              22: {'name': 'train_labeltoimage_AAFnew22_230809_172545', 'epoch': 300},
              23: {'name': 'train_labeltoimage_AAFnew23_230809_175240', 'epoch': 300},
              24: {'name': 'train_labeltoimage_AAFnew24_230809_181923', 'epoch': 300},
              25: {'name': 'train_labeltoimage_AAFnew25_230809_184611', 'epoch': 250},
              26: {'name': 'train_labeltoimage_AAFnew26_230809_191302', 'epoch': 200},
              27: {'name': 'train_labeltoimage_AAFnew27_230809_193948', 'epoch': 300},
              28: {'name': 'train_labeltoimage_AAFnew28_230809_200631', 'epoch': 50},
              29: {'name': 'train_labeltoimage_AAFnew29_230809_203313', 'epoch': 200},
              30: {'name': 'train_labeltoimage_AAFnew30_230809_205949', 'epoch': 200},
              31: {'name': 'train_labeltoimage_AAFnew31_230809_212621', 'epoch': 150},
              32: {'name': 'train_labeltoimage_AAFnew32_230809_215304', 'epoch': 150},
              33: {'name': 'train_labeltoimage_AAFnew33_230809_221943', 'epoch': 300},
              34: {'name': 'train_labeltoimage_AAFnew34_230809_224633', 'epoch': 200},
              35: {'name': 'train_labeltoimage_AAFnew35_230809_231312', 'epoch': 250},
              36: {'name': 'train_labeltoimage_AAFnew36_230809_233942', 'epoch': 300},
              37: {'name': 'train_labeltoimage_AAFnew37_230810_000613', 'epoch': 300},
              38: {'name': 'train_labeltoimage_AAFnew38_230810_003256', 'epoch': 300},
              39: {'name': 'train_labeltoimage_AAFnew39_230810_005942', 'epoch': 300},
              40: {'name': 'train_labeltoimage_AAFnew40_230809_163255', 'epoch': 250},
              41: {'name': 'train_labeltoimage_AAFnew41_230809_165933', 'epoch': 200},
              42: {'name': 'train_labeltoimage_AAFnew42_230809_172616', 'epoch': 300},
              43: {'name': 'train_labeltoimage_AAFnew43_230809_175257', 'epoch': 250},
              44: {'name': 'train_labeltoimage_AAFnew44_230809_181935', 'epoch': 300},
              45: {'name': 'train_labeltoimage_AAFnew45_230809_184620', 'epoch': 300},
              46: {'name': 'train_labeltoimage_AAFnew46_230809_191311', 'epoch': 300},
              47: {'name': 'train_labeltoimage_AAFnew47_230809_194003', 'epoch': 200},
              48: {'name': 'train_labeltoimage_AAFnew48_230809_200648', 'epoch': 250},
              49: {'name': 'train_labeltoimage_AAFnew49_230809_203326', 'epoch': 300},
              50: {'name': 'train_labeltoimage_AAFnew50_230809_210007', 'epoch': 100},
              51: {'name': 'train_labeltoimage_AAFnew51_230809_212655', 'epoch': 300},
              52: {'name': 'train_labeltoimage_AAFnew52_230809_215343', 'epoch': 200},
              53: {'name': 'train_labeltoimage_AAFnew53_230809_222029', 'epoch': 300},
              54: {'name': 'train_labeltoimage_AAFnew54_230809_224713', 'epoch': 150},
              55: {'name': 'train_labeltoimage_AAFnew55_230809_231401', 'epoch': 250},
              56: {'name': 'train_labeltoimage_AAFnew56_230809_234050', 'epoch': 300},
              57: {'name': 'train_labeltoimage_AAFnew57_230810_000742', 'epoch': 300},
              58: {'name': 'train_labeltoimage_AAFnew58_230810_003432', 'epoch': 100},
              59: {'name': 'train_labeltoimage_AAFnew59_230810_010130', 'epoch': 300},
              60: {'name': 'train_labeltoimage_AAFnew60_230809_163304', 'epoch': 250},
              61: {'name': 'train_labeltoimage_AAFnew61_230809_165951', 'epoch': 300},
              62: {'name': 'train_labeltoimage_AAFnew62_230809_172636', 'epoch': 250},
              63: {'name': 'train_labeltoimage_AAFnew63_230809_175331', 'epoch': 300},
              64: {'name': 'train_labeltoimage_AAFnew64_230809_182018', 'epoch': 250},
              65: {'name': 'train_labeltoimage_AAFnew65_230809_184718', 'epoch': 300},
              66: {'name': 'train_labeltoimage_AAFnew66_230809_191412', 'epoch': 250},
              67: {'name': 'train_labeltoimage_AAFnew67_230809_194112', 'epoch': 300},
              68: {'name': 'train_labeltoimage_AAFnew68_230809_200804', 'epoch': 200},
              69: {'name': 'train_labeltoimage_AAFnew69_230809_203453', 'epoch': 300},
              70: {'name': 'train_labeltoimage_AAFnew70_230809_210157', 'epoch': 250},
              71: {'name': 'train_labeltoimage_AAFnew71_230809_212852', 'epoch': 50},
              72: {'name': 'train_labeltoimage_AAFnew72_230809_215542', 'epoch': 200},
              73: {'name': 'train_labeltoimage_AAFnew73_230809_222238', 'epoch': 300},
              74: {'name': 'train_labeltoimage_AAFnew74_230809_224932', 'epoch': 300},
              75: {'name': 'train_labeltoimage_AAFnew75_230809_231634', 'epoch': 250},
              76: {'name': 'train_labeltoimage_AAFnew76_230809_234337', 'epoch': 250},
              77: {'name': 'train_labeltoimage_AAFnew77_230810_001035', 'epoch': 250},
              78: {'name': 'train_labeltoimage_AAFnew78_230810_003739', 'epoch': 250}}


def create_test_configs():
    config_dir = '/chenyongfeng/FedST/diffusion_model/Palette-Image-to-Image-Diffusion-Models-main-join/config'
    AAF_separate_config_dir = os.path.join(config_dir, 'AAFnew_join_test')
    os.makedirs(AAF_separate_config_dir, exist_ok=True)

    base_config = os.path.join(config_dir, 'AAFnew0_join_test.json')

    json_str = ''
    with open(base_config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt_base = json.loads(json_str, object_pairs_hook=OrderedDict)

    for style_id in tqdm(range(79)):
        target_style_dir = os.path.join(AAF_separate_config_dir, str(style_id))
        os.makedirs(target_style_dir, exist_ok=True)
        data_source_id = list(range(79))
        data_source_id.remove(style_id)
        for data_id in data_source_id:
            opt = copy.deepcopy(opt_base)
            opt['name'] = f"AAFnew_data{data_id}_style{style_id}"
            opt['datasets']['test']['which_dataset']['args'][
                'data_flist'] = f"datasets/AAF_Face_new/flist/test.flist{data_id}"
            opt['datasets']['test']['which_dataset']['args']['data_client'] = f"{data_id}"
            opt['datasets']['test']['which_dataset']['args']['model_client'] = f"{style_id}"
            opt['datasets']['test']['dataloader']['args']['batch_size'] = 16
            # 风格模型
            # opt['path']['resume_state'] = f"experiments/{model_dict[style_id]['name']}/checkpoint/{model_dict[style_id]['epoch']}"
            opt['path']['resume_state'] = "/chenyongfeng/FedST/dataset/AAF_Face_new/model_fedavg_fedst_ddpm_0812_023636/model50_folds0generator"

            save_path = os.path.join(target_style_dir, f'{opt["name"]}.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(opt, f, ensure_ascii=False, indent=4)


def create_sh(task_num=4, total=79, fake_style_num=3):
    task_num_for_each_gpu = (total // task_num) + (1 if total % task_num != 0 else 0)
    total_num = 0
    for task_id in range(task_num):
        sh_path = os.path.join(base_dir, f'test_AAFnew{task_id + 1}.sh')
        with open(sh_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n\n")
            f.write('echo "Start running train..."\n')
            for t in range(total_num, min(total_num + task_num_for_each_gpu, total)):
                f.write(f'echo "Style {t}..."\n')
                config_base_dir = os.path.join(base_dir, 'config', 'AAFnew_join_test')
                json_list = sorted(glob(os.path.join(config_base_dir, str(t), '*')))
                for json_path in json_list:
                    json_name = os.path.basename(json_path)
                    f.write(
                        f"python ./run.py -c config/AAFnew_join_test/{t}/{json_name} -p test --gpu_ids {task_id}\n")
            f.write('echo "Scripts have finished running."')
        total_num += task_num_for_each_gpu


def get_fake_image_ages(total=79, fake_style_num=3):
    client_partition = np.linspace(0, total, fake_style_num + 2, dtype=np.int64).tolist()
    age_interval_map = {}
    for age in range(total):
        for index in range(len(client_partition) - 1):
            if age < client_partition[index + 1]:
                age_interval_map[age] = index
                break
    print(age_interval_map)
    # 计算需要生成的fake_style_num个年龄的age_id
    interval_ids = list(range(fake_style_num + 1))
    fake_ages = {}

    for age in range(total):
        other_intervals = copy.deepcopy(interval_ids)
        other_intervals.remove(age_interval_map[age])
        delta =  age- client_partition[age_interval_map[age]]
        fake_ages[age] = []
        for interval_id in other_intervals:
            fake_ages[age].append(client_partition[interval_id] + delta)
    print(fake_ages)
    with open(os.path.join(base_dir,'fake_ages.json'), 'w', encoding='utf-8') as f:
        json.dump(fake_ages, f, ensure_ascii=False, indent=4)
    print('fake_ages.json saved')
    return fake_ages

def create_sh_by_fake_ages(task_num=4, total=79, fake_style_num=3):
    fake_ages_dict =get_fake_image_ages(total=total, fake_style_num=fake_style_num)
    task_num_for_each_gpu = (total // task_num) + (1 if total % task_num != 0 else 0)
    total_num = 0
    for task_id in range(task_num):

        sh_path = os.path.join(base_dir, f'test_AAFnew{task_id + 1}.sh')
        with open(sh_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n\n")
            f.write('echo "Start running train..."\n')
            # 对每个原始年龄按照fake_ages生成其他年龄的数据
            for age in range(total_num, min(total_num + task_num_for_each_gpu, total)):
                f.write(f'echo "Age {age}..."\n')
                config_base_dir = os.path.join(base_dir, 'config', 'AAFnew_join_test')
                fake_ages = fake_ages_dict[age]
                for fake_age in fake_ages:
                    json_name = f"AAFnew_data{age}_style{fake_age}.json"
                    f.write(
                        f"python ./run.py -c config/AAFnew_join_test/{fake_age}/{json_name} -p test --gpu_ids {task_id}\n")
            f.write('echo "Scripts have finished running."')
        total_num += task_num_for_each_gpu


if __name__ == '__main__':
    create_test_configs()
    # create_sh(task_num=4, total=79, fake_style_num=3)
    create_sh_by_fake_ages(task_num=4, total=79, fake_style_num=3)
