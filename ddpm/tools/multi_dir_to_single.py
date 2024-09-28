from glob import glob
from tqdm import tqdm
import os
import shutil


base_dir = '/root/data/AI3D/FedST/a_tools/AAF_face_new/real_image_with_fake'
target_dir = '/root/data/AI3D/FedST/a_tools/AAF_face_new/real_image_with_fake_gather'
# gather_client_ids = list(range(0,10))
# target_client_id = 0
# gather_client_ids = list(range(30,40))
# target_client_id = 1
gather_client_ids = list(range(60,70))
target_client_id = 2

for client_id in tqdm(gather_client_ids):
    image_paths = glob(os.path.join(base_dir, str(client_id), '*'))
    target_client_dir = os.path.join(target_dir,str(target_client_id))
    os.makedirs(target_client_dir,exist_ok=True)
    for image_path in image_paths:
        shutil.copy(image_path, os.path.join(target_client_dir, os.path.basename(image_path)))
