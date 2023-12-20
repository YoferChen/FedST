# from json.tool import main
# import sys
# import numpy as np
# import torch
import os
# from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
# from algorithm.fedavg.optim import FedAvg
# from algorithm.fedprox.optim import FedProx
# from algorithm.feddyn.optim import FedDyn
import sys

sys.path.insert(0, './')
import models.unet_model.networks as network_unet
from models.losses.losses import *
import copy
from models.fedst_ddpm_model.ddpm_models import ddpm_config
import torch.nn.functional as F

class FedSTDdpmModel(BaseModel):
    def name(self):
        return 'DdpmModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, d_seg_gt, d_seg_sim):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, d_seg_gt, d_seg_sim), flags) if f]

        return loss_filter

    def initialize(self, opt):
        self.opt = opt
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features

        ##### define networks        
        # Generator network
        self.ddpm_opt = ddpm_config.ddpm_opt
        self.ddpm_opt['model']['which_networks'][0]['args']['unet']['num_client'] = opt.client_num_in_total
        print("num_client = ", self.ddpm_opt['model']['which_networks'][0]['args']['unet']['num_client'])
        self.netG, self.criterionDiffusion = ddpm_config.get_netG_and_losses(self.ddpm_opt)
        self.netG = torch.nn.DataParallel(self.netG, device_ids=opt.gpu_ids)
        self.netG.to(0)

        if opt.pretrained_model is not None:
            import collections
            # state_dict = torch.load("models/fedst_ddpm_model/ddpm_models/weights/200_Network.pth", map_location='cuda:0')
            # state_dict = torch.load("models/fedst_ddpm_model/ddpm_models/weights/40_Network.pth", map_location='cuda:0')  # Pretrained DDPM model trained by Face dataset
            state_dict = torch.load(opt.pretrained_model_for_DDPM, map_location='cuda:0')

            sd = collections.OrderedDict()
            count = 0
            for i in self.netG.state_dict().keys():
                key = i.replace('module.', '')
                try:
                    sd[i] = state_dict[key]
                    count += 1
                except Exception as e:
                    sd[i] = self.netG.state_dict()[i]
                    print(f'Skip:{e}]')
            print('count=', count)
            self.netG.load_state_dict(sd, strict=True)
            print('[Pretrained DDPM Model] Pretrained model {} has loaded. '.format(opt.pretrained_model_for_DDPM))

        self.netG.module.set_loss(self.criterionDiffusion)
        self.netG.module.set_new_noise_schedule(phase="train")
        # Seg network
        # main_device = 2
        main_device = len(opt.gpu_ids) - 1  # Unet and ddpm are placed in the different GPU
        self.netDseg = network_unet.define_net(input_nc=opt.input_nc, output_nc=opt.output_nc, net=opt.net, \
                                               init_type=opt.init_type, init_gain=opt.init_gain,
                                               gpu_ids=copy.deepcopy(self.gpu_ids), main_device=main_device)
        # unet pretrained model
        if opt.pretrained_model_for_DDPM is not None:
            # pretrain_model_path = './pretrained_model/Face_Segment/model43_folds0_best.pkl'
            pretrain_model_path = opt.pretrained_model
            print('[Pretrained UNet Model for Training DDPM] Pretrain model for unet {} has loaded. '.format(
                pretrain_model_path))

            state_dict = torch.load(pretrain_model_path)
            sd_unet = collections.OrderedDict()
            count = 0
            for i in self.netDseg.state_dict().keys():
                key = i.replace('module.', 'net.module.')
                try:
                    sd_unet[i] = state_dict[key]
                    count += 1
                except Exception as e:
                    sd_unet[i] = self.netDseg.state_dict()[i]
                    print(f'Error:{e}]')
            print('count=', count)

            self.netDseg.load_state_dict(sd_unet, strict=True)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # # define loss functions
            self.criterionSeg = FocalLoss(alpha=opt.focal_alpha)

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())

            if opt.federated_algorithm == 'fedavg':
                self.optimizer_G = torch.optim.Adam(params=params, lr=opt.lr)
                params = list(self.netDseg.parameters())
                self.optimizer_Seg = torch.optim.Adam(params=params, lr=opt.lr)
            '''
            elif opt.federated_algorithm == 'fedprox':
                self.optimizer_G = FedProx(params,
                                           momentum=0.9,
                                           mu=opt.mu,
                                           gmf=opt.gmf,
                                           lr=opt.lr,
                                           nesterov=False,
                                           weight_decay=1e-4,
                                           alpha=0.9,
                                           eps=1e-5)
                params = list(self.netDseg.parameters())
                self.optimizer_Seg = FedProx(params,
                                             momentum=0.9,
                                             mu=opt.mu,
                                             gmf=opt.gmf,
                                             lr=opt.lr,
                                             nesterov=False,
                                             weight_decay=1e-4,
                                             # lr=opt.lr,
                                             alpha=0.9,
                                             eps=1e-5)
            elif opt.federated_algorithm == 'feddyn':
                self.optimizer_G = FedDyn(params,
                                          momentum=0.9,
                                          nesterov=False,
                                          weight_decay=1e-4,
                                          dyn_alpha=opt.dyn_alpha,
                                          lr=opt.lr,
                                          alpha=0.9,
                                          eps=1e-5)
                params = list(self.netDseg.parameters())
                self.optimizer_Seg = FedDyn(params,
                                            momentum=0.9,
                                            nesterov=False,
                                            weight_decay=1e-4,
                                            dyn_alpha=opt.dyn_alpha,
                                            lr=opt.lr,
                                            alpha=0.9,
                                            eps=1e-5)
            elif opt.federated_algorithm == 'feddc':
                self.optimizer_G = torch.optim.Adam(params=params, lr=opt.lr)
                params = list(self.netDseg.parameters())
                self.optimizer_Seg = torch.optim.Adam(params=params, lr=opt.lr)
        '''

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.output_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.output_nc, size[2], size[3])
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(label_map.device)
            input_label = input_label.scatter_(1, label_map.data.long().to(label_map.device), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.to(label_map.device)
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.to(label_map.device))

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.to(label_map.device))
            if self.opt.label_feat:
                inst_map = label_map.to(label_map.device)

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)

            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, image, resize_image, cond_image, class_, netG_constant, round=None):
        real_image = image
        resize_real_image = resize_image
        loss_G = self.netG(resize_real_image.cuda(0), cond_image.cuda(0), class_=class_.cuda(0))

        if round > 50:
            # Fake other Generation
            with torch.no_grad():
                class_other = []
                while (len(class_other) < len(class_)):
                    index = np.random.randint(self.opt.client_num_in_total)
                    if index != class_[0].item():
                        class_other.append(index)
                class_other = torch.tensor(class_other).type(torch.float32).cuda(0)
                netG_constant.module.set_new_noise_schedule(phase="test")
                fake_image_other = netG_constant.module.restoration(cond_image.cuda(0), class_=class_other)
                # fake_image_other 256 --> 384
                fake_image_other = F.interpolate(fake_image_other, size=(self.opt.fineSize, self.opt.fineSize))
                if self.opt.input_nc < 3:
                    # rgb to gray
                    fake_image_other = torch.mean(fake_image_other, dim=1, keepdim=True)

            # Seg loss
            # fake_seg_other = self.netDseg.forward(fake_image_other.detach().cuda(1))
            # real_seg = self.netDseg.forward(real_image.cuda(1))
            fake_seg_other = self.netDseg.forward(fake_image_other.detach().cuda(0))
            real_seg = self.netDseg.forward(real_image.cuda(0))
            loss_seg = 0.5 * (self.criterionSeg(real_seg, label.type(torch.int64)) + self.criterionSeg(fake_seg_other,
                                                                                                       label.type(
                                                                                                           torch.int64))) \
                       + 0.5 * (dice_loss(real_seg, label.type(torch.int64)) * 10 + dice_loss(fake_seg_other,
                                                                                              label.type(
                                                                                                  torch.int64)) * 10)
        else:
            fake_image_other = None
            # real_seg = self.netDseg.forward(real_image.cuda(1))
            real_seg = self.netDseg.forward(real_image.cuda(0))
            loss_seg = self.criterionSeg(real_seg, label.type(torch.int64)) + dice_loss(real_seg,
                                                                                        label.type(torch.int64)) * 10

        # Only return the fake_B image if necessary to save BW
        return [loss_G.cuda(0), loss_seg.cuda(0), fake_image_other]

    def gray_to_rgb(self, image):
        # 假设 batch 中的单通道图像为 input，维度为 [batch_size, 1, height, width]
        batch_size, _, height, width = image.shape

        # 创建全零张量作为 RGB 通道
        r_channel = torch.zeros(batch_size, 1, height, width)
        g_channel = torch.zeros(batch_size, 1, height, width)
        b_channel = torch.zeros(batch_size, 1, height, width)

        # 将原始单通道图像复制为三个通道的输入
        r_channel[:, 0, :, :] = image[:, 0, :, :]
        g_channel[:, 0, :, :] = image[:, 0, :, :]
        b_channel[:, 0, :, :] = image[:, 0, :, :]

        # 将 3 个通道的张量合并为一个三通道图像
        output = torch.cat([r_channel, g_channel, b_channel], dim=1)
        return output

    def inference(self, label, inst, image=None, class_=None):
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        if self.use_features:
            if self.opt.use_encoded_image:
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat, class_)
        else:
            fake_image = self.netG.forward(input_concat, class_)
        return fake_image

    def train(self):
        self.netDseg.train()
        self.netG.train()

    def eval(self):
        self.netDseg.eval()
        self.netG.eval()

    def seg_inference(self, label, inst, image, feat):
        input_label, _, real_image, _ = self.encode_input(label, inst, image, feat)
        [b, c, m, n] = input_label.shape
        label = torch.zeros([b, m, n]).to(input_label.device)
        label[input_label[:, 1, :, :] == 1] = 1
        seg = self.netDseg.forward(real_image)
        if label is not None:
            loss_D_seg_gt = self.criterionSeg(seg, label.type(torch.int64)) + dice_loss(seg,
                                                                                        label.type(torch.int64)) * 10
        else:
            loss_D_seg_gt = None
        return seg, loss_D_seg_gt

    def sample_features(self, inst):
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.output_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netDseg, 'Dseg', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(net='diffusion_model')
        return parser
