from __future__ import print_function
import torch
import numpy as np
import skimage.morphology as sm
from PIL import Image
import os
import logging
from logging import handlers
import matplotlib.pyplot as plt
import copy
from skimage.io import imread, imsave
#import xlsxwriter
 
#生成excel文件
def generate_excel(expenses, list_title):
    workbook = xlsxwriter.Workbook('./rec_data.xlsx')
    worksheet = workbook.add_worksheet()
 
    # 设定格式，等号左边格式名称自定义，字典中格式为指定选项
    # bold：加粗，num_format:数字格式
    bold_format = workbook.add_format({'bold': True})
    #money_format = workbook.add_format({'num_format': '$#,##0'})
    #date_format = workbook.add_format({'num_format': 'mmmm d yyyy'})
 
    # 将二行二列设置宽度为15(从0开始)
    worksheet.set_column(1, 1, 15)
 
    # 用符号标记位置，例如：A列1行
    for key in list_title.keys():
        worksheet.write(list_title[key], key, bold_format)

    row = 1
    col = 0
    for item in (expenses):
        bias = 0
        # 使用write_string方法，指定数据格式写入数据
        for key in list_title.keys():
            worksheet.write_string(row, col+bias, str(item[key]))
            bias += 1
        row += 1
    workbook.close()
    
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)
# Converts a output tensor into a gray label map
def post_process(output_tensor,no_closing): #b,c,h,w->h,w
    out = torch.softmax(output_tensor[0],0)
    out_numpy = out.max(0)[1].cpu().numpy()
    out_numpy = np.array(out_numpy).astype('uint8')
    if not no_closing:
        out_numpy=sm.closing(out_numpy,sm.square(3))
    out_numpy_ske=sm.skeletonize(out_numpy)#骨架化
    out_numpy_ske=out_numpy_ske.astype('uint8')
    results={'out':out_numpy,'out_ske':out_numpy_ske}
    return results

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8, std=0.5, mean=0.5,nc=3):
    if isinstance(input_image, torch.Tensor):
        image_numpy = input_image[0].cpu().float().numpy()
    else:
        image_numpy = input_image
    if len(image_numpy.shape) == 2:#h,w->1,h,w
        image_numpy = image_numpy[None,:,:]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))*std+mean)*255.0#3,h,w->h,w,3
    if nc == 1:
        image_numpy = image_numpy[..., 0] * 0.299 + image_numpy[...,1] * 0.587 + image_numpy[...,2] * 0.114 #h,w,3->h,w
    return image_numpy.astype(imtype)



###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    print(label_tensor)
    label_tensor = label_tensor.cpu().float()
    label_tensor = label_tensor.float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
        lt = copy.deepcopy(label_tensor)
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    if image_numpy.dtype == np.float32:
        image_numpy = 255*image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

        
def adjust_learning_rate(learning_rate, epoch, opt):
    """Sets the learning rate to the initial LR decayed by half every 10 epochs until 1e-6"""
    lr = learning_rate * (opt.lr_gamma ** (epoch // 2))
    if(lr < 1e-6):
        lr = 1e-6
        
    return lr


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        # for name, param in exp_mdl.named_parameters():
        #     n_par += len(param.data.reshape(-1))
        for param_key in exp_mdl.state_dict():
            n_par += len(exp_mdl.state_dict()[param_key].data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        # for name, param in mdl.named_parameters():
        #     temp = param.data.cpu().numpy().reshape(-1)
        #     param_mat[i, idx:idx + len(temp)] = temp
        #     idx += len(temp)
        for param_key in mdl.state_dict():
            temp = mdl.state_dict()[param_key].data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def set_client_from_params(mdl, params):
    # dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    dict_param = copy.deepcopy(mdl.state_dict())
    idx = 0
    # for name, param in mdl.named_parameters():
    #     weights = param.data
    #     length = len(weights.reshape(-1))
    #     dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to("cuda"))
    #     idx += length
    for param_key in mdl.state_dict():
        weights = mdl.state_dict()[param_key].data
        length = len(weights.reshape(-1))
        dict_param[param_key].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to("cuda"))
        idx += length

    mdl.load_state_dict(dict_param)

    return mdl