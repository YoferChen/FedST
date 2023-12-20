import numpy as np
import gala.evaluate as ev
from skimage.measure import label
import time
import pdb


# ---------------for evaluation----------------------
def calculate_vi_ri_ari(result, gt):
    # false merges(缺失), false splits（划痕）
    merger_error, split_error = ev.split_vi(result, gt)
    vi = merger_error + split_error
    ri = ev.rand_index(result, gt)
    adjust_ri = ev.adj_rand_index(result, gt)
    return {'vi': vi, 'ri': ri, 'adjust_ri': adjust_ri,
            'merger_error': merger_error,
            'split_error': split_error}


def calculate_ap(label_pred, num_pred, label_mask, num_mask):
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)
    count = 0
    m_iou = 0
    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))  # 获得与之相交的所有label
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:  # 如果pred的某一个label没有与之对应的mask的label,则继续下一个label
            continue
        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))

        iou = intersect_mask_label_area / (union_mask_label_area)
        max_iou = np.max(iou, axis=0)
        m_iou += max_iou
        count += 1
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    fp = num_pred - tp
    fn = num_mask - tp
    map_score = np.average(tp / (tp + fp + fn))
    m_iou /= (count + 1e-5)
    return {'map_score': map_score, 'm_iou': m_iou}


def get_map_miou_vi_ri_ari(pred, mask, boundary=255):
    """
    map https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    :param pred: Predict Mask，[0, 255], Foreground 255, background 0
    :param mask: Ground True Mask，[0, 255]，Foreground 255, background 0
    :return: map F VI RI aRI
    """
    # pdb.set_trace()
    try:
        assert np.shape(pred) == np.shape(mask)
    except:
        pdb.set_trace()
    # 1px边缘闭合
    '''
    pred[[0,-1], :] = boundary
    pred[:,[0,-1]] = boundary
    mask[[0,-1], :] = boundary
    mask[:, [0,-1]] = boundary
    '''
    time1 = time.time()
    '''
    material need convert to label formate
    '''
    # label_mask, num_mask = label(mask, connectivity=1, background= boundary, return_num=True)
    # label_pred, num_pred = label(pred, connectivity=1, background= boundary, return_num=True)
    '''
    other wise 
    '''
    label_mask = mask
    label_pred = pred
    num_mask, num_pred = 19, 19
    results = {}
    m_ap_iou = calculate_ap(label_pred, num_pred, label_mask, num_mask)
    results.update(m_ap_iou)
    vi_ri = calculate_vi_ri_ari(label_pred, label_mask)
    results.update(vi_ri)
    return results
