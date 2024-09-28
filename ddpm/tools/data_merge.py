from glob import glob
import shutil
import os
from tqdm import tqdm

# data_base = '/root/data/AI3D/FedST/data_dir/selected'
# target_dir = '/root/data/AI3D/FedST/data_dir/selected_total'
data_base = '../data_dir/selected_resize'
target_dir = '../data_dir/selected_resize_total'

data_dirs = glob(data_base+'/*')
for data_dir in tqdm(data_dirs):
    os.makedirs(target_dir, exist_ok=True)
    image_paths = glob(os.path.join(data_dir, '*'))
    for image_path in image_paths:
        target_path = os.path.join(target_dir, os.path.basename(image_path))
        shutil.copy(image_path, target_path)