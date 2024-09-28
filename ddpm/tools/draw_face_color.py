from PIL import Image
import numpy as np


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


def get_color_image(mask_path: str, color_path: str):
    AAF_ids = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
               'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'beard']
    AAF_full_name_ids = ['background', 'skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'eye_glass',
                         'left_ear',
                         'right_ear', 'ear_ring', 'nose', 'mouth', 'upper_lip', 'lower_lip', 'neck', 'neck_lace',
                         'cloth', 'hair',
                         'hat', 'beard']
    CelebAMask_ids = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                      'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth', 'beard']

    # 将AAF的id映射到CelebAMask的id
    maps = {}
    for i in range(20):
        new_id = CelebAMask_ids.index(AAF_ids[i])
        maps[i] = new_id
    print(maps)
    mask = np.array(Image.open(mask_path))
    color_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colors = get_pascal_labels()
    for i in range(1, 20):
        color_image[:, :, 0][mask == i] = colors[maps[i]][0]
        color_image[:, :, 1][mask == i] = colors[maps[i]][1]
        color_image[:, :, 2][mask == i] = colors[maps[i]][2]
    Image.fromarray(color_image).save(color_path)
    return color_image


def get_color_image_for_array(mask: np.ndarray, color_path: str, map_id=False):
    '''
    mask: (H, W)
    '''
    AAF_ids = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
               'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'beard']
    AAF_full_name_ids = ['background', 'skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'eye_glass',
                         'left_ear',
                         'right_ear', 'ear_ring', 'nose', 'mouth', 'upper_lip', 'lower_lip', 'neck', 'neck_lace',
                         'cloth', 'hair',
                         'hat', 'beard']
    CelebAMask_ids = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear',
                      'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth', 'beard']

    # 将AAF的id映射到CelebAMask的id
    maps = {}
    for i in range(20):
        if map_id:
            new_id = CelebAMask_ids.index(AAF_ids[i])
            maps[i] = new_id
        else:
            maps[i] = i
    color_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colors = get_pascal_labels()
    for i in range(1, 20):
        color_image[:, :, 0][mask == i] = colors[maps[i]][0]
        color_image[:, :, 1][mask == i] = colors[maps[i]][1]
        color_image[:, :, 2][mask == i] = colors[maps[i]][2]
    Image.fromarray(color_image).save(color_path)
    return color_image


if __name__ == '__main__':
    mask_path = r'C:\Users\17325\wisdom_store_workspace\Local\FedST_Face_align\data\00007A02.png'
    color_path = r'C:\Users\17325\wisdom_store_workspace\Local\FedST_Face_align\data\00007A02_color.png'
    get_color_image(mask_path, color_path)
