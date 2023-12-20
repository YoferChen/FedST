import pdb
# from cv2 import moments
import torch
import numpy as np
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
# from fedml_api.standalone.fedprox.optim import FedProx
# from fedml_api.standalone.scaffold.optim import Scaffold
# from fedml_api.standalone.feddyn.optim import FedDyn
# from fedml_api.standalone.fedavg.optim import FedAvg
from algorithm.fedavg.optim import FedAvg
from algorithm.fedprox.optim import FedProx
from algorithm.feddyn.optim import FedDyn
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
from collections import OrderedDict
from ..losses.losses import *
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

log_dir = "./logs"
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)
a_count = 1


class UnetModel(BaseModel):
    def name(self):
        return 'UnetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(net='unet')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['seg']
        self.model_names = ['']
        self.visual_names = ['image', 'out']
        opt.net = 'unet'
        self.net = networks.define_net(input_nc=opt.input_nc, output_nc=opt.output_nc, net=opt.net, \
                                       init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.visual_names.append('label')
            if opt.federated_algorithm == 'fedavg':
                self.optimizer = FedAvg(self.net.parameters(),  # 正式实验
                                        lr=opt.lr,
                                        alpha=0,
                                        momentum=0,
                                        eps=1e-5)
                # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr)  # 平时跑模型

            elif opt.federated_algorithm == 'fedprox':
                self.optimizer = FedProx(self.net.parameters(),
                                         lr=opt.lr,
                                         mu=0.0001,
                                         # gmf = opt.gmf,
                                         momentum=0,
                                         nesterov=False,
                                         weight_decay=1e-4,
                                         alpha=0,
                                         eps=1e-5)
            elif opt.federated_algorithm == 'feddyn':
                self.optimizer = FedDyn(self.net.parameters(),
                                        lr=opt.lr,
                                        momentum=0,
                                        nesterov=False,
                                        weight_decay=1e-4,
                                        dyn_alpha=0.0001,
                                        alpha=0,
                                        eps=1e-5)
            elif opt.federated_algorithm == 'feddc':
                self.optimizer = torch.optim.SGD(self.net.parameters(),
                                                     lr=opt.lr,
                                                     weight_decay=1e-4)
            elif opt.federated_algorithm == 'fedddpm':
                self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr)
            else:
                self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=opt.lr, momentum=0.9)
            self.optimizers.append(self.optimizer)
        # weight = None
        # if isinstance(opt.loss_weight,list):
        #     weight=torch.Tensor(opt.loss_weight)
        #     weight=weight.to(self.gpu_ids[0])
        self.criterion = networks.define_loss(opt.loss_type, opt.focal_alpha, opt.output_nc)
        temperature = 0.05
        self.cont_loss_func = losses.NTXentLoss(temperature)

    def set_input(self, input):  # input(image,label,image_path)

        self.image = input['image']  # .to(self.gpu_ids[0])
        # if self.isTrain:
        self.label = input['label'].squeeze(1).type(torch.LongTensor)  # .to(self.gpu_ids[0])
        self.images_path = input['path']
        if self.opt.federated_algorithm == 'fedddpm':
            self.fake_image = input['fake_image'].to(self.gpu_ids[0])
            # self.fake_label = input['fake_label'].squeeze(1).type(torch.LongTensor).to(self.gpu_ids[0])
            # self.images = torch.cat((self.image, self.fake_image), dim=0)
            # self.labels = torch.cat((self.label, self.label), dim=0)
            # global a_count
            # writer.add_images("image",
            #                   self.images,
            #                   a_count, dataformats="NCHW")
            # writer.add_images("label",
            #                   torch.cat((input['label'], input['fake_label']), dim=0),
            #                   a_count, dataformats="NCHW")
            # # writer.add_images("fake_image",
            # #                   self.fake_image,
            # #                   a_count, dataformats="NCHW")
            # # writer.add_images("fake_label",
            # #                   input['fake_label'],
            # #                   a_count, dataformats="NCHW")
            # a_count = a_count + 1

    def forward(self):
        self.out = self.net(self.image)
        if np.isnan(np.sum(self.out.detach().cpu().numpy())):
            a = 1
        return self.out

    def cal_loss(self):
        self.loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        return self.loss_seg.item()

    def backword(self):
        # pdb.set_trace()
        l = self.cal_loss()
        if np.isnan(l):
            a = 1
        try:
            self.loss_seg.backward()
        except:
            a = 1
        return l

    def optimize_parameters(self):
        self.set_requires_grad(self.net, True)
        self.forward()
        self.optimizers[0].zero_grad()
        self.backword()

        # clip grad
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
        if self.opt.federated_algorithm != 'feddyn':
            self.optimizers[0].step()

    def fedddpm_optimize_parameters(self):
        self.set_requires_grad(self.net, True)
        # forward
        # self.out = self.net(self.images[0].to(self.gpu_ids[0]))
        # out_fake = self.net(self.images[1].to(self.gpu_ids[0]))
        self.out = self.net(self.image.to(self.gpu_ids[0]))
        out_fake = self.net(self.fake_image.to(self.gpu_ids[0]))
        # compute loss
        # loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        loss_seg = 0.5 * (self.criterion(self.out, self.label) + self.criterion(out_fake, self.label)) \
                   + 0.5 * (dice_loss(self.out, self.label) * 10 + dice_loss(out_fake, self.label) * 10)
        # loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        # zero grad
        self.optimizers[0].zero_grad()
        # backward
        loss_seg.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
        self.optimizers[0].step()

        # # train fake
        # self.fake_out = self.net(self.fake_image.to(self.gpu_ids[0]))
        # loss_seg = self.criterion(self.fake_out, self.label) + dice_loss(self.fake_out, self.label) * 10
        # self.optimizers[0].zero_grad()
        # loss_seg.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
        # self.optimizers[0].step()

    def feddc_optimize_parameters(self, alpha, local_update_last, global_update_last, global_model_param, hist_i):
        self.set_requires_grad(self.net, True)
        # print('data: ', np.mean(self.image.cpu().numpy()), np.var(self.image.cpu().numpy()))
        self.forward()
        loss_seg = self.criterion(self.out, self.label) + dice_loss(self.out, self.label) * 10
        ## Get f_i estimate
        loss_f_i = loss_seg

        state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32, device="cuda:0")

        local_parameter = None
        # for param in self.net.parameters():
        #     if not isinstance(local_parameter, torch.Tensor):
        #         # Initially nothing to concatenate
        #         local_parameter = param.reshape(-1)
        #     else:
        #         local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
        for param_keys in self.net.state_dict():
            if not isinstance(local_parameter, torch.Tensor):
                # Initially nothing to concatenate
                local_parameter = self.net.state_dict()[param_keys].reshape(-1)
            else:
                local_parameter = torch.cat((local_parameter, self.net.state_dict()[param_keys].reshape(-1)), 0)
        loss_cp = alpha / 2 * torch.sum(
            (local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))
        loss_cg = torch.sum(local_parameter * state_update_diff)

        loss = loss_f_i + loss_cp + loss_cg
        self.optimizers[0].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.net.parameters(),
                                       max_norm=10)  # Clip gradients to prevent exploding
        self.optimizers[0].step()

    def set_learning_rate(self, lr):
        for param_group in self.optimizers[0].param_groups:
            param_group['lr'] = lr

    def extract_contour_embedding(self, contour_list, embeddings):
        average_embeddings_list = []
        for contour in contour_list:
            contour = contour.to(embeddings.device)
            contour_embeddings = contour * embeddings
            average_embeddings = torch.sum(contour_embeddings, (-1, -2)) / torch.sum(contour, (-1, -2))
            # print (contour.shape)
            # print (embeddings.shape)
            # print (contour_embeddings.shape)
            # print (average_embeddings.shape)
            average_embeddings_list.append(average_embeddings)
        return average_embeddings_list
