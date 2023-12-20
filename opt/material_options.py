import argparse
import torch
import os
import pdb
import sys
sys.path.append('./')
import models as models
import data as data


from util import util
class BaseOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):

        parser.add_argument('--gpu_ids',type=str,default='0,1,2',help='gpu ids:e.g.0 0,1,2 1,2 use -1 for CPU')
        parser.add_argument('--name', type=str, default = 'material' , required=False, help='name of the experiment.')
        parser.add_argument('--model',type=str,default='fedst',choices=['unet','fedst', 'fedst_ddpm'],help='chooses which model to use,unet...')
        parser.add_argument('--federated_algrithom',type=str,default='fedavg',choices=['fedavg', 'fedprox', 'feddyn', 'feddc', 'fedddpm'],help='chooses which federated learning algrithom to use')
        parser.add_argument('--net',type=str,default='unet',choices=['unet'],help='chooses which network to use,unet...')
        parser.add_argument('--init_type', type=str, default='xavier_uniform', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=1.0, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix')
        parser.add_argument('--checkpoints_dir',type=str,default='./model_checkpoints',help='models are saved here')   
        parser.add_argument('--log_dir', default='fedst.log', type=str, help='customized suffix: opt.name = opt.name + suffix')

        parser.add_argument('--epochs', type=int, default=1, metavar='EP',help='how many epochs will be trained locally')
        parser.add_argument('--client_num_in_total', type=int, default=2, metavar='NN',help='number of workers in a distributed cluster')
        parser.add_argument('--client_num_per_round', type=int, default=2, metavar='NN',help='number of workers')
        parser.add_argument('--comm_round', type=int, default=50, help='how many round of communications we shoud use')
        parser.add_argument('--fineSize', type=int, default=384,help='if_wandb')
        parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')

       
        parser.add_argument('--dataroot',default= './dataset/Material',help='path to images')
        parser.add_argument('--resultroot',default= './ckpt_material',help='path to ckpt')
        parser.add_argument('--data_type',default='splited', help='[pure|mix|splited]')

        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        parser.add_argument('--serial_batches', default=False, action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_normalize', action='store_true',help='do not use normalization')

        parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [aligned,gan]')

        parser.add_argument('--batch_size',type=int,default = 6, help='input batch size')
        parser.add_argument('--input_nc',type=int,default = 1, help='input image channels')
        parser.add_argument('--output_nc',type=int,default= 2, help='output classes')
        
        #loss
        parser.add_argument('--loss_type', choices=['CrossEntropyLoss', 'BCW', 'Focal'], default='Focal', help='loss types')
        #[0.0013, 0.0363, 0.3248, 0.3188, 0.3188]
        parser.add_argument('--focal_alpha', type=list, default = [0.1, 0.9], help='alpha of focal loss')

        
        ##pix2pixHD
        parser.add_argument('--no_flip', type=bool, default=True,help='if_wandb')
        parser.add_argument('--loadSize', type=int, default=384,help='if_wandb')
        parser.add_argument('--no_instance', type=bool, default=True,help='if_wandb')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        #gennerator
        parser.add_argument('--instance_feat', type=bool, default=False, help='alpah parameter in feddyn')
        parser.add_argument('--label_feat', type=bool, default=False, help='alpah parameter in feddyn')
        parser.add_argument('--netG', type=str, default='local', help='selects model to use for netG')
        parser.add_argument('--ngf', type=int, default=48, help='# of gen filters in first conv layer')
        parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        parser.add_argument('--n_blocks_global', type=int, default=6, help='number of residual blocks in the global generator network')
        parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer') 
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')   
        parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location') 
        #discraminator
        parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

        parser.add_argument('--dyn_alpha', type=int, default=0.0001, help='Feddyn optimize param alpha')
        self.initialized = True
        return parser
    
    def gather_options(self):
    # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        
        model_option_setter = models.get_option_setter(model_name)

        parser = model_option_setter(parser, self.isTrain)

        opt, _ = parser.parse_known_args()  # parse again with the new default

        self.parser = parser

        return parser.parse_args(args=[])
    
    def parse(self):
        
        opt=self.gather_options()
        opt.isTrain = self.isTrain  # train or test
        if opt.net in ['unet11','unet16','albunet']:
            opt.input_nc = 3
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        
        
            
        #set gpu ids
        gpu_ids=opt.gpu_ids.split(',')
        opt.gpu_ids=[]
        for str_id in gpu_ids:
            id=int(str_id)
            if id >=0:
                opt.gpu_ids.append(id)
        #set cuda device
        if len(opt.gpu_ids)>0:
            torch.cuda.set_device(opt.gpu_ids[0])
            
        self.opt = opt 
        #self.print_options(opt)
        return opt
    
    
    def print_options(self,opt):
        message=''
        message+='---------Options---------\n'
        for k,v in sorted(vars(opt).items()):
            comment=''
            default= self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        #save to the disk
        if self.isTrain:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
                
    
    
class TrainOptions(BaseOptions):
    def initialize(self,parser):
        parser = BaseOptions.initialize(self, parser)
        
        # for training
        parser.add_argument('--cross_validation', type=bool, default=True, help='if using cross validation to choose the model')
        parser.add_argument('--folds', type=int, default=5, help='how many folds in cross_validation')
        parser.add_argument('--total_img_dir_real', type=str, default='train/real_image', help='where are the training images')
        parser.add_argument('--total_label_dir_real', type=str, default='train/real_label', help='where are the training images')
        parser.add_argument('--no_eval', action='store_true', help='no eval in training')
        parser.add_argument('--val_img_list', type=str, help='val images name list')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial lr')
        parser.add_argument('--lr_gamma', type=float, default=0.927, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain=True
        return parser


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--test_img_dir', type=str, default='test/real_image', help='where are the test images')
        parser.add_argument('--test_label_dir', type=str,default='test/real_label', help='where are the test labels')
        parser.add_argument('--results_dir', type=str, default='./results/', help='save results here')
        parser.add_argument('--boundary', type=int, default=255, help='boundary mask 255|0')
        self.isTrain = False
        return parser

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options