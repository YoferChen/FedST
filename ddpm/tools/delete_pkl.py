from glob import glob
import os
from tqdm import tqdm


base_dir = '/chenyongfeng/FedST/diffusion_model/Palette-Image-to-Image-Diffusion-Models-main/experiments/0808'

dirs = glob(os.path.join(base_dir, '*'))
for train_dir in tqdm(dirs):
    if 'train_labeltoimage_AAF' in os.path.basename(train_dir):
        print(train_dir)

# dir_names = ['train_labeltoimage_AAF21_230807_044903',
#              'train_labeltoimage_AAF22_230807_052744',
#              'train_labeltoimage_AAF23_230807_060628',
#              'train_labeltoimage_AAF24_230807_064328',
#              'train_labeltoimage_AAF25_230807_072110']

        data_dir = os.path.join(train_dir, 'checkpoint')
        files = glob(os.path.join(data_dir, '*'))
        for file in files:
            filename = os.path.basename(file)
            round_index = filename.split('_')[0]
            round_index = round_index.split('.')[0]
            if int(round_index)%50!=0:
                print(f'Delete: {file}')
                os.remove(file)