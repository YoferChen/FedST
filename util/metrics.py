import math
import numpy as np
from skimage import metrics
from sklearn.metrics import f1_score
from skimage.measure import label
import gala.evaluate as ev
import pydicom
from skimage.io import imsave
import numpy as np
import glob
# import cv2
from scipy import ndimage
from sklearn.neighbors import KDTree


# 我们将图像分割指标分成5个类别：
# 比如：基于像素的，基于类内重合度的，基于边缘的，基于聚类的 和 基于实例的
# We grouped image segmentation metrics into five groups:
# Such as pixel based, region based, boundary based, clustering based, instance based

# 注意：
# 对于下列所有方法，pred是分割结果，mask是真值，所有像素的值为整数，且在[0,C], C为分割类别
# Note:
# For all the metrics below, pred is the segmentation result of certain method and
# mask is the ground truth. The value is integer and range from [0, C], where C is
# class of segmentation


def get_metric(metric_name: str, output: np.array, label: np.array) -> float:
    """
        Evaluate output and label and get metric value during training
    """
    temp_metric = 0.0
    if metric_name == 'dice':
        temp_metric = get_dice(output, label)
    elif metric_name == 'vi':
        _, _, temp_metric = get_vi(output, label)
    return temp_metric


def get_total_evaluation(pred: np.ndarray, mask: np.ndarray, require_edge: bool = True, bg_value=0):
    """
         Get whole evaluation of all metrics
         Return Tuple, (value_list, name_list)
    """
    metric_values = {}
    gala = get_vi(pred, mask)
    if require_edge:
        metric_values['pa'] = get_pixel_accuracy(pred, mask)
        metric_values['mpa'] = get_mean_accuracy(pred, mask)
        metric_values['mprecision'] = get_mean_precision(pred, mask)
        metric_values['mrecall'] = get_mean_recall(pred, mask)
        metric_values['miou'] = get_iou(pred, mask)
        metric_values['fwiou'] = get_F1_score(pred, mask)
        metric_values['f1-score'] = get_F1_score(pred, mask)
        metric_values['dice'], metric_values['ravd'] = evaluate(Vref=mask, Vseg=pred)
        metric_values['vi'] = get_vi(pred, mask, bg_value=bg_value)[-1]
        metric_values['ari'] = get_ari(pred, mask, bg_value=bg_value)
        metric_values['ap50'] = get_ap50(pred, mask, bg_value=bg_value)
        metric_values['map'] = get_map_2018kdsb(pred, mask, bg_value=bg_value)
        metric_values['rel_grain_size'] = get_related_grain_size(pred, mask)
    else:
        metric_values = [get_pixel_accuracy(pred, mask), get_mean_accuracy(pred, mask),
                         get_iou(pred, mask), get_fwiou(pred, mask), get_dice(pred, mask),
                         get_ri(pred, mask), get_ari(pred, mask), gala[0], gala[1], gala[2],
                         get_cardinality_difference(pred, mask), get_map_2018kdsb(pred, mask),
                         get_vi(pred, mask), get_ari(pred, mask)]
    return metric_values


# ************** 基于像素的评估 Pixel based evaluation **************
# pixel accuracy, mean accuracy
def get_pixel_accuracy(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Pixel accuracy for whole image
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1

    temp_n_ii = 0.0
    temp_t_i = 0.0
    for i_cl in range(class_num):
        temp_n_ii += np.count_nonzero(mask[pred == i_cl] == i_cl)
        temp_t_i += np.count_nonzero(mask == i_cl)
    value = temp_n_ii / temp_t_i
    return value


def get_mean_accuracy(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean accuracy for each class
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1
    temp = 0.0
    for i_cl in range(class_num):
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl) + 1e-20
        temp += n_ii / t_i
    value = temp / class_num
    return value


def get_mean_precision(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    查准率 = TP/(TP+FP)
    Mean precision for each class
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    # class_num = np.amax(mask) + 1
    classes = np.unique(mask)
    temp = 0.0
    # for i_cl in range(class_num):
    for i_cl in classes:
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(pred == i_cl) + 1e-20
        temp += n_ii / t_i
    # value = temp / class_num
    value = temp / len(classes)
    return value


def get_mean_recall(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    召回率 = TP/(TP+FN)
    Mean recall for each class
    Referenced by：
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    # class_num = np.amax(mask) + 1
    classes = np.unique(mask)
    temp = 0.0
    # for i_cl in range(class_num):
    for i_cl in classes:
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl) + 1e-20
        temp += n_ii / t_i
    # value = temp / class_num
    value = temp / len(classes)
    return value


def get_ap50(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    modified by 浅若清风cyf
    """
    # thresholds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    thresholds = np.array([0.5])
    # tp = np.zeros(10)
    tp = np.zeros(1)

    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)

    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))  # 获得与之相交的所有label
        # 对与其相交的的所有mask label计算iou，后取其最值
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:  # 如果pred的某一个label没有与之对应的mask的label,则继续下一个label
            continue

        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))
        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        # 根据最值将tp赋值
        # Assumption: There is only a region whose IOU > 0.5 for target region
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    fp = num_pred - tp
    fn = num_mask - tp
    value = np.average(tp / (tp + fp + fn))
    return value


def get_F1_score(pred: np.ndarray, mask: np.ndarray) -> float:
    value = f1_score((pred).reshape(-1), (mask).reshape(-1), average='macro')
    return value


# ************** 基于类内重合度的评估 Region based evaluation **************
# Mean IOU (mIOU), Frequency weighted IOU(FwIOU), Dice score
def get_iou(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Referenced by:
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1

    temp = 0.0
    for i_cl in range(class_num):
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl)
        temp += n_ii / (t_i + np.count_nonzero(pred == i_cl) - n_ii + 1e-20)
    value = temp / class_num
    return value


def get_fwiou(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Referenced by:
    Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
    IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
    """
    class_num = np.amax(mask) + 1

    temp_t_i = 0.0
    temp_iou = 0.0
    for i_cl in range(0, class_num):
        n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
        t_i = np.count_nonzero(mask == i_cl)
        temp_iou += t_i * n_ii / (t_i + np.count_nonzero(pred == i_cl) - n_ii + 1e-20)
        temp_t_i += t_i
    value = temp_iou / temp_t_i
    return value


def get_dice(pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Dice score
    From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
    """
    intersection = np.count_nonzero(mask[pred == 1] == 1)
    area_sum = np.count_nonzero(mask == 1) + np.count_nonzero(pred == 1)
    value = 2 * intersection / area_sum
    return value


# ************** 基于边缘的评估 boundary based evaluation **************
# figure of merit
def get_figure_of_merit(pred: np.ndarray, mask: np.ndarray, boundary_value: int = 0, const_index: float = 0.1) -> float:
    """
    Referenced by:
    Abdou I E, Pratt W K. Quantitative design and evaluation of enhancement thresholding edge detectors[J].
    Proceedings of the IEEE, 1979, 67(5): 753-763
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    num_max = num_pred if num_pred > num_mask else num_mask
    temp = 0.0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                temp = temp + 1 / (1 + const_index * pow(distance, 2))
    f_score = (1.0 / num_max) * temp
    return f_score


def get_dis_from_mask_point(mask: np.ndarray, index_x: int, index_y: int, boundary_value: int = 0,
                            neighbor_length: int = 20):
    """
    Calculation the distance between the boundary point(pred) and its nearest boundary point(mask)
    """

    if mask[index_x, index_y] == 255:
        return 0
    distance = neighbor_length / 2
    region_start_row = 0
    region_start_col = 0
    region_end_row = mask.shape[0]
    region_end_col = mask.shape[1]
    if index_x - neighbor_length > 0:
        region_start_row = index_x - neighbor_length
    if index_x + neighbor_length < mask.shape[0]:
        region_end_row = index_x + neighbor_length
    if index_y - neighbor_length > 0:
        region_start_col = index_y - neighbor_length
    if index_y + neighbor_length < mask.shape[1]:
        region_end_col = index_y + neighbor_length
        # Get the corrdinate of mask in neighbor region
        # becuase the corrdinate will be chaneged after slice operation, we add it manually
    x, y = np.where(mask[region_start_row: region_end_row, region_start_col: region_end_col] == boundary_value)
    try:
        min_distance = np.amin(
            np.linalg.norm(np.array([x + region_start_row, y + region_start_col]) - np.array([[index_x], [index_y]]),
                           axis=0))
        return min_distance
    except ValueError as e:
        return neighbor_length


# completeness
def get_completeness(pred: np.ndarray, mask: np.ndarray, theta: float = 2.0, boundary_value: int = 0) -> float:
    """
    Referenced by:
    Beyond the Pixel-Wise Loss for Topology-Aware Delineation
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    temp_pred_mask = 0
    for index_x in range(0, mask.shape[0]):
        for index_y in range(0, mask.shape[1]):
            if mask[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(pred, index_x, index_y)
                if distance < theta:
                    temp_pred_mask += 1
    f_score = float(temp_pred_mask) / float(num_mask)
    return f_score


# correctness
def get_correctness(pred: np.ndarray, mask: np.ndarray, theta: float = 2.0, boundary_value: int = 0) -> float:
    """
    Referenced by:
    Beyond the Pixel-Wise Loss for Topology-Aware Delineation
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    temp_mask_pred = 0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                if distance < theta:
                    temp_mask_pred += 1
    f_score = float(temp_mask_pred) / float(num_pred)
    return f_score


# quality
def get_quality(pred: np.ndarray, mask: np.ndarray, theta: float = 2.0, boundary_value: int = 0) -> float:
    """
    Referenced by:
    Beyond the Pixel-Wise Loss for Topology-Aware Delineation
    """
    num_pred = np.count_nonzero(pred == boundary_value)
    num_mask = np.count_nonzero(mask == boundary_value)
    temp_pred_mask = 0
    for index_x in range(0, mask.shape[0]):
        for index_y in range(0, mask.shape[1]):
            if mask[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(pred, index_x, index_y)
                if distance < theta:
                    temp_pred_mask += 1
    temp_mask_pred = 0
    for index_x in range(0, pred.shape[0]):
        for index_y in range(0, pred.shape[1]):
            if pred[index_x, index_y] == boundary_value:
                distance = get_dis_from_mask_point(mask, index_x, index_y)
                if distance < theta:
                    temp_mask_pred += 1
    f_score = float(temp_mask_pred) / float(num_pred - temp_pred_mask + num_mask)
    return f_score


# ************** 基于聚类的评估 Clustering based evaluation **************
# Rand Index (RI), Adjusted Rand Index (ARI) and Variation of Information (VI)
def get_ri(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    Rand index
    Implemented by gala (https://github.com/janelia-flyem/gala.)
    """
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    value = ev.rand_index(label_pred, label_mask)
    return value


def get_ari(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    Adjusted rand index
    Implemented by gala (https://github.com/janelia-flyem/gala.)
    """
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    value = ev.adj_rand_index(label_pred, label_mask)
    return value


def get_vi(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0, method: int = 1):
    """
    Referenced by:
    Marina Meilă (2007), Comparing clusterings—an information based distance,
    Journal of Multivariate Analysis, Volume 98, Issue 5, Pages 873-895, ISSN 0047-259X, DOI:10.1016/j.jmva.2006.11.013.
    :param method: 0: skimage implementation and 1: gala implementation (https://github.com/janelia-flyem/gala.)
    :return Tuple = (VI, merger_error, split_error)
    """
    vi, merger_error, split_error = 0.0, 0.0, 0.0

    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    if method == 0:
        # scikit-image
        split_error, merger_error = metrics.variation_of_information(label_mask, label_pred)
    elif method == 1:
        # gala
        merger_error, split_error = ev.split_vi(label_pred, label_mask)
    vi = merger_error + split_error
    if math.isnan(vi):
        return 10, 5, 5
    return merger_error, split_error, vi


# ************** 基于实例的评估 Instance based evaluation **************
# cardinality difference, MAP
def get_cardinality_difference(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
    R = |G| - |P|
    |G| is number of region in mask, and |P| is number of region in pred
    R > 0 refers to under segmentation and R < 0 refers to over segmentation
    Referenced by
    Waggoner J , Zhou Y , Simmons J , et al. 3D Materials Image Segmentation by 2D Propagation: A Graph-Cut Approach Considering Homomorphism[J].
    IEEE Transactions on Image Processing, 2013, 22(12):5282-5293.
    """

    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    value = num_mask - num_pred
    return value * 1.0


def get_map_2018kdsb(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    """
    Mean Average Precision
    From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
    Referenced from 2018 kaggle data science bowl:
    https://www.kaggle.com/c/data-science-bowl-2018/overview/evaluation
    """
    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    tp = np.zeros(10)

    label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
    label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
    # label_mask, num_mask = mask,19
    # label_pred, num_pred= pred,19
    for i_pred in range(1, num_pred + 1):
        intersect_mask_labels = list(np.unique(label_mask[label_pred == i_pred]))  # 获得与之相交的所有label
        # 对与其相交的的所有mask label计算iou，后取其最值
        if 0 in intersect_mask_labels:
            intersect_mask_labels.remove(0)

        if len(intersect_mask_labels) == 0:  # 如果pred的某一个label没有与之对应的mask的label,则继续下一个label
            continue

        intersect_mask_label_area = np.zeros((len(intersect_mask_labels), 1))
        union_mask_label_area = np.zeros((len(intersect_mask_labels), 1))

        for index, i_mask in enumerate(intersect_mask_labels):
            intersect_mask_label_area[index, 0] = np.count_nonzero(label_pred[label_mask == i_mask] == i_pred)
            union_mask_label_area[index, 0] = np.count_nonzero((label_mask == i_mask) | (label_pred == i_pred))
        iou = intersect_mask_label_area / union_mask_label_area
        max_iou = np.max(iou, axis=0)
        # 根据最值将tp赋值
        # Assumption: There is only a region whose IOU > 0.5 for target region
        tp[thresholds < max_iou] = tp[thresholds < max_iou] + 1
    fp = num_pred - tp
    fn = num_mask - tp
    value = np.average(tp / (tp + fp + fn))
    return value


def get_grain_size(img, M=100.0):
    """
    计算奥氏体平均晶粒度G
    M：放大倍数
    L：截线长度
    P_avg:平均截点数
    计算公式 G = 6.643856 * lg((M*P_avg)/L) - 3.288
    return:奥氏体平均晶粒度G
    """
    row_length = img.shape[0]  # 宽
    col_length = img.shape[1]  # 长
    #     print(row_length, col_length)
    # L=0.1094um*5520 = 603.888um
    # Todo:lack of the resolution and scale of super_alloy!!!!
    L = 1
    # L = row_length
    list_row = []
    # 三条截线初始位置(这里是竖着划线，截断长的边
    for i in range(3):
        temp = row_length * (i + 1) * 0.25
        list_row.append(int(temp))
    # print(list_row)
    # 获取截点
    P = [0, 0, 0]
    for i in range(3):
        count = 0
        # print(i, len(img[list_row[i]]))
        # print(img[list_row[i]].shape)
        for j in img[list_row[i]]:
            # if img[list_row[i]][current][0] != img[list_row[i]][current + 1][0]:
            # print(j)
            # 截断长边的时候
            # if img[current][list_row[i]] != img[current + 1][list_row[i]]:
            # 截断短边的时候
            if img[list_row[i]][count] != img[list_row[i]][count + 1]:
                P[i] += 1
            count += 1
            # print(current)
            if count == col_length - 1:
                break
    # print(current, "done!")

    P_avg = (P[0] + P[1] + P[2]) / (3 * 2)
    P_avg = P_avg if P_avg != 0 else 1e-4

    #  计算晶粒度G
    G = 6.643856 * math.log((M * P_avg) / L, 10) - 3.288
    # G = decimal.Decimal(G).quantize(decimal.Decimal("0.0001"))
    #     print("计算得到的晶粒度：", G)
    return math.fabs(G)


# cardinality difference, MAP
def get_related_grain_size(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
    value = abs(get_grain_size(mask) - get_grain_size(pred)) / get_grain_size(mask)
    return value * 1.0


def evaluate(Vref, Vseg, dicom_dir=None):
    dice = DICE(Vref, Vseg)
    ravd = RAVD(Vref, Vseg)
    # [assd, mssd]=SSD(Vref,Vseg,dicom_dir)
    return dice, ravd


def DICE(Vref, Vseg):
    dice = 2 * (Vref & Vseg).sum() / (Vref.sum() + Vseg.sum() + 1e-5)
    return dice


def RAVD(Vref, Vseg):
    ravd = (abs(Vref.sum() - Vseg.sum()) / Vref.sum()) * 100
    return ravd


def SSD(Vref, Vseg, dicom_dir):
    struct = ndimage.generate_binary_structure(3, 1)

    ref_border = Vref ^ ndimage.binary_erosion(Vref, structure=struct, border_value=1)
    ref_border_voxels = np.array(np.where(ref_border))

    seg_border = Vseg ^ ndimage.binary_erosion(Vseg, structure=struct, border_value=1)
    seg_border_voxels = np.array(np.where(seg_border))

    ref_border_voxels_real = transformToRealCoordinates(ref_border_voxels, dicom_dir)
    seg_border_voxels_real = transformToRealCoordinates(seg_border_voxels, dicom_dir)

    tree_ref = KDTree(np.array(ref_border_voxels_real))
    dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real)
    tree_seg = KDTree(np.array(seg_border_voxels_real))
    dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real)

    assd = (dist_seg_to_ref.sum() + dist_ref_to_seg.sum()) / (len(dist_seg_to_ref) + len(dist_ref_to_seg))
    mssd = np.concatenate((dist_seg_to_ref, dist_ref_to_seg)).max()
    return assd, mssd


def transformToRealCoordinates(indexPoints, dicom_dir):
    """
    This function transforms index points to the real world coordinates
    according to DICOM Patient-Based Coordinate System
    The source: DICOM PS3.3 2019a - Information Object Definitions page 499.
    
    In CHAOS challenge the orientation of the slices is determined by order
    of image names NOT by position tags in DICOM files. If you need to use
    real orientation data mentioned in DICOM, you may consider to use
    TransformIndexToPhysicalPoint() function from SimpleITK library.
    """

    dicom_file_list = glob.glob(dicom_dir + '/*.dcm')
    dicom_file_list.sort()
    # Read position and orientation info from first image
    ds_first = pydicom.dcmread(dicom_file_list[0])
    img_pos_first = list(map(float, list(ds_first.ImagePositionPatient)))
    img_or = list(map(float, list(ds_first.ImageOrientationPatient)))
    pix_space = list(map(float, list(ds_first.PixelSpacing)))
    # Read position info from first image from last image
    ds_last = pydicom.dcmread(dicom_file_list[-1])
    img_pos_last = list(map(float, list(ds_last.ImagePositionPatient)))

    T1 = img_pos_first
    TN = img_pos_last
    X = img_or[:3]
    Y = img_or[3:]
    deltaI = pix_space[0]
    deltaJ = pix_space[1]
    N = len(dicom_file_list)
    M = np.array([[X[0] * deltaI, Y[0] * deltaJ, (T1[0] - TN[0]) / (1 - N), T1[0]],
                  [X[1] * deltaI, Y[1] * deltaJ, (T1[1] - TN[1]) / (1 - N), T1[1]],
                  [X[2] * deltaI, Y[2] * deltaJ, (T1[2] - TN[2]) / (1 - N), T1[2]], [0, 0, 0, 1]])

    realPoints = []
    for i in range(len(indexPoints[0])):
        P = np.array([indexPoints[1, i], indexPoints[2, i], indexPoints[0, i], 1])
        R = np.matmul(M, P)
        realPoints.append(R[0:3])

    return realPoints
