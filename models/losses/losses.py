import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
import pdb

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred, target):
        if pred.dim()>2:
            pred = pred.view(pred.size(0),pred.size(1),-1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1,2)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1,pred.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).to(pred.device)
        logpt = F.log_softmax(pred, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt) ** self.gamma * logpt * 100
        if self.size_average: return loss.mean()
        else: return loss.sum()


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    Copied from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss
        
class WeightMapLoss(nn.Module):
    """
    calculate weighted loss with weight maps
    """
    def __init__(self, class_num=2, average=True, eps = 1e-20):
        super(WeightMapLoss, self).__init__()
        self.class_num = class_num
        self.average = average
        self.eps = eps
            
    def forward(self, pred, weight_maps):
        """
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        method：Select the type of loss function
        """
        #pdb.set_trace()
        logit = torch.softmax(pred, dim=1)
        weight_maps = weight_maps.float().to(logit.device)
        loss = -1 * weight_maps * (torch.log(logit + self.eps))
        if np.isnan((loss.sum() / weight_maps.sum()).item()):
            a = 1
        #loss = -1 * weight_maps * (torch.log(logit + 1e-60))
        if self.average: return loss.sum() / (weight_maps.sum()+ self.eps)
        else: return loss.sum()

class BalancedClassWeight():
    """
    Balanced Class Weight
    If ratio of classes of image is [n1:n2:n3...:nm], m is class number, and the sum is 1
    So the weight is [1 - n1, 1 - n2, 1 - n3, ..., 1 - nm]
    """
    def __init__(self, class_num = 2):
        self._class_num = class_num
    
    def _get_weight(self, label):
        #pdb.set_trace()
        total_num = label.size
        weight = np.zeros((label.shape[0], label.shape[1], self._class_num))
        class_weight = np.zeros((self._class_num, 1))
        for idx in range(self._class_num):
            idx_num = np.count_nonzero(label == idx)
            class_weight[idx, 0] = idx_num
        class_weight = (total_num - class_weight * 1.0) / total_num
        for idx in range(self._class_num):
            weight[:, :, idx][label == idx] = class_weight[idx, 0]
        return weight
        
    def get_weight(self, label: np.ndarray) -> np.ndarray:
        weight = np.zeros((label.shape[0], label.shape[1], self._class_num))
        class_weight = np.zeros((self._class_num, 1))
        for idx in range(self._class_num):
            idx_num = np.count_nonzero(label == idx)
            class_weight[idx, 0] = idx_num
        t_matrix = class_weight[class_weight != 0]
        if t_matrix.size > 1:
            min_num = np.amin(t_matrix)
            # min_num = np.amin(class_weight)
            class_weight = class_weight * 1.0 / min_num
            class_weight = np.sum(class_weight) - class_weight
            for idx in range(self._class_num):
                weight[:, :, idx][label == idx] = class_weight[idx, 0]
        else:  # t_matrix.size == 1
            weight[:, :, np.argwhere(class_weight != 0)[0, 0]] = 1
        return weight

    def _get_weight_tensor(self, label):
        #pdb.set_trace()
        batch,w,h = label.shape
        weight = torch.zeros((batch, self._class_num,w, h)).to(label.device)
        for b in range(batch):
            total_num = w*h
            class_weight = torch.zeros((self._class_num, 1))
            #pdb.set_trace()
            for idx in range(self._class_num):
                idx_num = torch.sum(label[b,:,:] == idx)
                class_weight[idx, 0] = idx_num
            class_weight = ((total_num - class_weight * 1.0) / total_num).to(label.device)
            for idx in range(self._class_num):
                weight[b, idx, :, :][label[b,:,:] == idx] = class_weight[idx, 0]
        return weight

class DisTransWeight():
    """
    Distance Transform Weighted Map
    """
    def __init__(self, class_num = 2):
        self._class_num = class_num
    
    def _get_weight(self, label):
        weight = np.zeros((label.shape[0], label.shape[1], self._class_num))
        for idx in range(self._class_num):
            temp = np.zeros_like(label)
            temp[label == idx] = 1
            dis_weight = distance(temp)
            weight[:, :, idx] = dis_weight
        return weight

class BCWLoss(nn.Module):
    
    def __init__(self, class_num = 2):
        super(BCWLoss, self).__init__()
        self.bcw = BalancedClassWeight(class_num = class_num)
        self.wm_loss = WeightMapLoss()
        
    def forward(self, pred, label):
        bcw_map = self.bcw._get_weight_tensor(label)
        if bcw_map.sum()==0:
             bcw_map = self.bcw._get_weight_tensor(label)
        loss = self.wm_loss(pred, bcw_map)
        return loss

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

def dice_loss(score, target):
    if len(target.shape) == 3:
        target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
    target = make_one_hot(target, score.shape[1]).to(score.device)
    target = target.float()
    score = F.sigmoid(score)
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss