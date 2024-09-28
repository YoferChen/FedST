from glob import glob
import os
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image


def get_map_dict():
    '''
    mask: (H, W)
    '''
    # AAF_ids = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    #            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'beard']
    AAF_ids = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
               'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'hair']  # beard to hair
    CelebAMask_ids = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                      'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth', 'beard']
    # 将AAF的id映射到CelebAMask的id
    maps = {}
    for i in range(20):
        new_id = CelebAMask_ids.index(AAF_ids[i])
        maps[i] = new_id
    return maps


# id_map = {0: 0, 1: 1, 2: 6, 3: 7, 4: 4, 5: 5, 6: 3, 7: 8, 8: 9, 9: 15, 10: 2, 11: 10, 12: 11, 13: 12, 14: 17, 15: 16, 16: 18, 17: 13, 18: 14, 19: 13}

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def get_color_image(mask: np.ndarray, num_classes: int):
    colors = get_pascal_labels()
    color_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(1, num_classes):
        color_image[:, :, 0][mask == i] = colors[i][0]
        color_image[:, :, 1][mask == i] = colors[i][1]
        color_image[:, :, 2][mask == i] = colors[i][2]
    return color_image


def dataset_map(data_dir: str, target_dir: str, id_map: dict):
    for mode in ['train', 'test']:
        for image_type in ['real_image', 'real_label']:
            print('Processing {} {}...'.format(mode, image_type))
            client_base_dir = os.path.join(data_dir, mode, image_type)
            client_dir_names = os.listdir(client_base_dir)
            for client_name in tqdm(client_dir_names):
                src_client_dir = os.path.join(client_base_dir, client_name)
                target_client_dir = os.path.join(target_dir, mode, image_type, client_name)
                target_client_color_dir = os.path.join(target_dir, mode, 'real_color', client_name)
                os.makedirs(target_client_dir, exist_ok=True)
                os.makedirs(target_client_color_dir, exist_ok=True)
                image_paths = glob(os.path.join(src_client_dir, '*'))
                for image_path in image_paths:
                    target_path = os.path.join(target_client_dir, os.path.basename(image_path))
                    color_path = os.path.join(target_client_color_dir, os.path.basename(image_path))
                    if image_type == 'real_image':
                        # 原图直接复制
                        shutil.copy(image_path, target_path)
                    elif image_type == 'real_label':
                        # mask图需要映射
                        mask = np.array(Image.open(image_path))
                        mask_new = np.zeros_like(mask, dtype=np.uint8)
                        for old_id in id_map:
                            mask_new[mask == old_id] = id_map[old_id]
                        Image.fromarray(mask_new).save(target_path)
                        # 根据映射后的mask生成color mask
                        color_mask = get_color_image(mask_new, len(id_map))
                        Image.fromarray(color_mask).save(color_path)


if __name__ == '__main__':
    id_map = get_map_dict()
    print(id_map)
    data_dir = '../dataset/AAF_Face_resplit'
    target_dir = '../dataset/AAF_Face_new'
    dataset_map(data_dir, target_dir, id_map)
