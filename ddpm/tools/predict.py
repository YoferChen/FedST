import argparse
import os
from skimage.io import imread
#from typing import OrderedDict
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as tr
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "cvpr_code/dataset")))
from torch.cuda.amp import autocast
parser = argparse.ArgumentParser()
# CHAOS_liver   Material   Face_segment
parser.add_argument('--dataset', default='face', choices=['material', 'face', 'medical', 'medical_material'], help='whitch dataset to use')
parser.add_argument('--model_path', type=str, default='/tanjing/cvpr_code/dataset/Face_segment/model_fedddpm/model40_folds', help='chooses which model to use,unet...')
parser.add_argument('--fold', type=str, default='0', help='chooses fold...')
parser.add_argument('--algrithom', type=str, default='fedddpm', help='chooses which federated learning algrithom to use')
op = parser.parse_args()
dataset = op.dataset
if dataset == 'material':
    from opt.material_options import TrainOptions, TestOptions
elif dataset == 'face':
    from opt.face_options import TrainOptions, TestOptions
elif dataset == 'medical':
    from opt.medical_options import TrainOptions, TestOptions
elif op.dataset == 'medical_material':
    from opt.medical_material_options import TrainOptions, TestOptions
fold = op.fold
fed_methods = op.algrithom
model_path = op.model_path+fold+'.pkl'
method = dataset+'_'+fed_methods+ '_fold_' + fold
#import opt.unet_options as op
from data.dataloader.data_loader import CreateDataLoader
from models import create_model 
import time
from util.util import post_process, mkdir, save_image, generate_excel
from util.evaluation import get_map_miou_vi_ri_ari
from util.metrics import get_total_evaluation
import util
import pdb 
import torch
#import cv2
from skimage.io import imsave
import tqdm


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    '''
    eval_results={}
    count=0
    t = 0
    t1 = 0
    root = "/yinxiang/Experiment_copy_modified/FedDG-ELCFS-main/epi_result/prediction_chaos_inner_dice"
    gt_path = sorted([i for i in os.listdir(root) if 'gt' in i ])
    pred_path = sorted([i for i in os.listdir(root) if '_1' in i ])
    for i, (data_p, label_p) in enumerate(zip(gt_path, pred_path)):
        pred = imread(os.path.join(root, data_p))
        mask = imread(os.path.join(root, label_p))
        eval_result = get_total_evaluation((pred/255.0).astype(np.int), (mask/255.0).astype(np.int))
        print(eval_result)
        count += 1
        message = ''
        for k,v in eval_result.items():
            if k in eval_results:
                eval_results[k]+=v
            else:
                eval_results[k]=v
            #message+='%s: %.5f\t'%(k,v)
    messge='total %d:\n'%count
    for k,v in eval_results.items():
        message += 'm_%s: %.5f\t' % (k, v/count)
    print(message)
    '''

    opt = TestOptions().parse()
    opt_train = TrainOptions().parse()
    if 'feddg' in fed_methods:
        opt_train.federated_algorithm = fed_methods
        opt.federated_algorithm = fed_methods
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip

    if opt.no_normalize:
        transform = tr.ToTensor()
    else:
        transform=tr.Compose([tr.ToTensor(),
                              tr.Normalize(mean=0.5,
                                           std=0.5)
                              ])
    #opt.dataroot = "/yinxiang/Experiment_copy_modified/FedDG-ELCFS-main/epi_result/prediction"
    #pdb.set_trace()
    opt.dataroot = '/root/data/AI3D/FedST/data_dir'
    opt.test_img_dir = '/root/data/AI3D/FedST/data_dir/selected/total'
    opt.test_label_dir = '/root/data/AI3D/FedST/data_dir/selected/total'
    data_loader = CreateDataLoader(opt,dataroot=opt.dataroot,image_dir=opt.test_img_dir,\
                                   label_dir=opt.test_label_dir,record_txt=None,transform=transform,is_aug=False)

    dataset = data_loader.load_data()
    datasize = len(data_loader)
    print('#test images = %d, batchsize = %d' %(datasize,opt.batch_size))
    #/yinxiang/Experiment_copy_modified/data/CHAOS_liver/model_fedavg/model28_folds0_best.pkl
    #
    #/yinxiang/Experiment_copy_modified/data/GlobalData/Total/model_fedprox/model_4_folds3.pkl
    #/yinxiang/Experiment_copy_modified/data/GlobalData/Total/model_fedprox/model_4_folds2.pkl
    #/yinxiang/Experiment_copy_modified/data/GlobalData/Total/model_fedprox/model_4_folds1.pkl
    #/yinxiang/Experiment_copy_modified/data/GlobalData/Total/model_fedprox/model_3_folds4.pkl
    #m_pa: 0.81907   m_mpa: 0.87096  m_miou: 0.53087 m_fwiou: 0.63527        m_f1-score: 0.63527     m_dice: 0.37832 m_ravd: 1534.24329 m_vi: 0.57062   m_ari: 0.21436  m_map: 0.00053
    #m_pa: 0.81884   m_mpa: 0.87072  m_miou: 0.53093 m_fwiou: 0.63532        m_f1-score: 0.63532     m_dice: 0.37868 m_ravd: 1535.79761      m_vi: 0.55669   m_ari: 0.21703        m_map: 0.00053
    #m_pa: 0.97982   m_mpa: 0.96590  m_miou: 0.83685 m_fwiou: 0.88674        m_f1-score: 0.88674     m_dice: 0.78429 m_ravd: 262.62031  m_vi: 0.00462   m_ari: 0.76254  m_map: 0.17226
    #m_pa: 0.97668   m_mpa: 0.96003  m_miou: 0.82085 m_fwiou: 0.87393        m_f1-score: 0.87393     m_dice: 0.76036 m_ravd: 266.88398       m_vi: 0.00635   m_ari: 0.73465        m_map: 0.10549
    #/yinxiang/Experiment_copy_modified/data/CHAOS_liver/model_fedavg_/model99_folds4.pkl
    state_dict = torch.load(model_path, map_location='cuda:0')
    #state_dict = torch.load('/yinxiang/Experiment_copy_modified/data/CHAOS_liver_aug/model_fedavg/model4_folds0.pkl')
    #state_dict = torch.load('/yinxiang/Experiment_copy_modified/data/CHAOS_liver/model_fedavg/model3_folds0.pkl')
    sd = OrderedDict()

    #opt.isTrain = True
    opt_train.model = 'unet'
    model = create_model(opt_train)
    # model.net = torch.nn.DataParallel(model.net, device_ids = [0,1])
    # pdb.set_trace()
    #model.setup(opt)
    if 'fedst' in method:
        for i in state_dict.keys():
            if 'netDseg' in i :
                key = '.'.join(['net']+i.split('.')[1:])
                #key = i
                sd[key] = state_dict[i]
        model.load_state_dict(sd)
    else:
        model.load_state_dict(state_dict)
    # model.load_state_dict(sd)
    #model.eval()
    
    img_dir = os.path.join(opt.results_dir, opt.name, '%s' %method)
    mkdir(img_dir)
    
    eval_results={}
    count=0
    with open(img_dir+'_eval.txt','w') as log:
        now = time.strftime('%c')
        log.write('=============Evaluation (%s)=============\n' % now)
    # test with eval mode.


    result = []
    

    for i, data in enumerate(tqdm.tqdm(dataset)):
        c_idx = 0
        #pdb.set_trace()
        # if i >= opt.num_test:
        #     break
        #pdb.set_trace()
        #data['label'][0,0,[0,-1],:] = 1
        #data['label'][0,0,:,[0,-1]] = 1
        #model.set_input(data)
        '''
        with torch.no_grad():
            losses, generated,fake_seg, real_seg  = model(Variable(data['label']).cuda(), Variable(data['inst']).cuda(), 
                    Variable(data['image']).cuda(), Variable(data['feat']).cuda(), Variable(data['class']).cuda(), model.netG)
        '''
        model.set_input(data)
        with torch.no_grad():
            pred = model()[0]

        
        #visuals = model.get_current_visuals()

        #post process skeletonize
        #pred = post_process(visuals['out'],opt.no_postprocess)['out']*255
        #pred = visuals['out'][0]
        
        pred = pred.max(0)[1].cpu().numpy()
        img_path = data['path'][0]
        short_path = os.path.basename(img_path)
        name = os.path.splitext(short_path)[0]
        image_name = '%s.png' % name  
        #plt.imshow(pred_numpy)
        #plt.imshow(pred)
        #plt.savefig(os.path.join(img_dir, image_name))
        #cv2.imwrite(os.path.join(img_dir, image_name), pred)
        #save_image(pred_numpy, )

        eval_start = time.time()
        count+=1
        #mask = data['label'].squeeze().numpy().astype(np.int)
        #plt.imshow(mask)
        #plt.savefig(os.path.join(img_dir, image_name.split('.')[0]+'_m.png'))
        # material need  * 255
        # mask = (data['label'].squeeze().numpy()*255).astype(np.uint8)
        mask = data['label'].squeeze().numpy().astype(np.uint8)
        #cv2.imwrite(os.path.join(img_dir, image_name.split('.')[0]+'_m.png'), mask)
        #eval_result=get_map_miou_vi_ri_ari(pred,mask,boundary=opt.boundary)
        eval_result = get_total_evaluation(pred,mask)
        #pred[pred==1]=255
        #mask[mask==1]=255
        #imsave(os.path.join(img_dir, image_name), pred.astype(np.uint8))
        #imsave(os.path.join(img_dir, image_name.split('.')[0]+'_m.png'), mask) 
        #print(visuals['out'].data[0].shape)
        print(data['label'].data[0].shape)
        

        
        # try:
            # imsave(os.path.join(img_dir, str(i)+'_'+image_name.split('.')[0]+'_1.png'), pred)
            # imsave(os.path.join(img_dir, str(i)+'_'+image_name.split('.')[0]+'_2.png'), 1 - pred)
            #imsave(os.path.join(img_dir, str(i)+'_'+image_name), util.util.tensor2label( mask, 19))
            # imsave(os.path.join(img_dir, str(i)+'_'+image_name.split('.')[0]+'_m.png'), util.util.tensor2label(data['label'].data[0], 19))
            # imsave(os.path.join(img_dir, str(i)+'_'+image_name.split('.')[0]+'_pred.png'), util.util.tensor2label(torch.from_numpy(pred).unsqueeze(0), 19))
            # imsave(os.path.join(img_dir, str(i)+'_'+image_name.split('.')[0]+'_m1.png'), data['label'].data[0][0].detach().numpy())
            # imsave(os.path.join(img_dir, str(i)+'_'+image_name.split('.')[0]+'_m2.png'), 1-data['label'].data[0][0].detach().numpy())
            # fake_0 = visuals['fake_image'][0][0].cpu().numpy()
            # fake_0 = fake_0*opt.transform_std[0]+opt.transform_mean[0]
            # fake_1 = visuals['fake_image2'][0][0].cpu().numpy()
            # fake_1 = fake_1*opt.transform_std[0]+opt.transform_mean[0]
            # imsave(os.path.join(img_dir, image_name.split('.')[0]+'_fake_0.png'), fake_0)
            # imsave(os.path.join(img_dir, image_name.split('.')[0]+'_fake_1.png'), fake_1)
        # except:
        #     pass
        message='%04d: %s \t'%(count,name)
        
        for k,v in eval_result.items():
            if k in eval_results:
                eval_results[k]+=v
            else:
                eval_results[k]=v
            message+='%s: %.5f\t'%(k,v)

            
        eval_result['name'] = name
        result.append(eval_result)
        #pdb.set_trace()
        print(message,'cost: %.4f'%(time.time()-eval_start))
        with open(img_dir + '_eval.txt', 'a') as log:
            log.write(message+'\n')
        
            
    messge='total %d:\n'%count
    for k,v in eval_results.items():
        message += 'm_%s: %.5f\t' % (k, v/count)
    print(message)
    with open(img_dir + '_eval.txt', 'a') as log:
        log.write(message+'\n')
    '''
    generate_excel(result, {'name':'A1', 'map_score':'B1', 
                            'vi':'C1', 'ri':'D1', 'adjust_ri':'E1',
                            'merger_error':'F1', 'split_error':'G1'})
    '''