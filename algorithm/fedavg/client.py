import logging
from torch.autograd import Variable
import torch
import sys
from torch import nn
import numpy as np
from skimage.io import imsave
from util.evaluation import get_map_miou_vi_ri_ari
from util.util import post_process, mkdir, save_image, adjust_learning_rate
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os
import pdb
import tqdm
import string
import copy
import thop


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, model, log):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.log = log
        self.args = args
        self.model = model

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def update_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self, w_global, round_idx):
        self.model.load_state_dict(w_global)
        # self.model.to(self.device)
        losses = {}
        epoch_loss = []
        self.model.train()
        test_data = self.local_test_data
        self.lr = adjust_learning_rate(self.args.lr, round_idx, self.args)
        # self.lr  = self.args.lr
        self.log.logger.info('lr : ' + str(self.lr))
        for epoch in range(self.args.epochs):
            epoch_iter = 0
            batch_loss = []
            # self.lr = self.lr*0.5**epoch
            for i, data in tqdm.tqdm(enumerate(self.local_training_data)):
                self.model.set_input(data)
                self.model.set_learning_rate(self.lr)
                if self.args.federated_algorithm == 'fedddpm':
                    self.model.fedddpm_optimize_parameters()
                else:
                    self.model.optimize_parameters()
                losses['train_loss'] = self.model.cal_loss()
                batch_loss.append(losses['train_loss'])
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # local eval
        self.model.eval()
        losses = {}
        epoch_loss_t = []
        batch_loss_t = []
        for i, data in enumerate(test_data):
            self.model.set_input(data)
            self.model.eval()
            self.model.test()
            losses['train_loss'] = self.model.cal_loss()
            batch_loss_t.append(losses['train_loss'])
        epoch_loss_t.append(sum(batch_loss_t) / len(batch_loss_t))
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_t) / len(epoch_loss_t)

    def train_ddpm(self, w_global, round_idx):
        self.model.load_state_dict(w_global)
        net_const = copy.deepcopy(self.model.netG).cuda(0)

        epoch_loss = []
        self.model.train()
        test_data = self.local_test_data
        self.lr = adjust_learning_rate(self.args.lr, round_idx, self.args)
        # self.lr = self.args.lr
        self.log.logger.info('lr : ' + str(self.lr))

        for epoch in range(self.args.epochs):
            batch_loss = []
            # self.lr = self.lr*0.5**epoch
            for i, data in tqdm.tqdm(enumerate(self.local_training_data)):
                loss_G, loss_Seg, fake_image_other = self.model(Variable(data['label']).cuda(),
                                                                Variable(data['image']).cuda(),
                                                                Variable(data['resize_img']).cuda(),
                                                                Variable(data['cond_image']).cuda(),
                                                                Variable(data['class']).cuda(),
                                                                net_const, round_idx)
                loss_G = torch.mean(loss_G)
                # print("loss_G = ", loss_G.item())
                loss_Seg = torch.mean(loss_Seg)

                ############### Backward Pass ####################
                # update generator weights
                self.model.optimizer_G.zero_grad()
                for p in self.model.optimizer_G.param_groups:
                    p['lr'] = self.lr
                loss_G.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.netG.parameters(),1)
                self.model.optimizer_G.step()
                # update segmentor weights
                self.model.optimizer_Seg.zero_grad()
                for p in self.model.optimizer_Seg.param_groups:
                    p['lr'] = self.lr
                loss_Seg.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.netDseg.parameters(),1)
                self.model.optimizer_Seg.step()
                if not np.isnan(loss_Seg.item()):
                    batch_loss.append(loss_Seg.item())
                else:
                    a = 0

            if fake_image_other != None:
                dim = fake_image_other.shape[1]
                if dim > 1:
                    # generated = generated[0].detach().cpu().numpy().transpose(1, 2, 0)
                    fake_image_other = fake_image_other[0].detach().cpu().numpy().transpose(1, 2, 0)
                else:
                    # generated = generated[0][0].detach().cpu().numpy()
                    fake_image_other = fake_image_other[0][0].detach().cpu().numpy()
                try:
                    imsave(os.path.join(self.args.dataroot + '/model_fedst_ddpm',
                                        str(self.client_idx) + '_' + str(round_idx) + '_generated_fake.png'),
                           fake_image_other)
                except:
                    pass
        if len(batch_loss) == 0:
            epoch_loss.append(9999)
        else:
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # local eval
        self.model.eval()
        epoch_loss_t = []
        batch_loss_t = []
        with torch.no_grad():
            for i, data in enumerate(test_data):
                _, loss = self.model.seg_inference(Variable(data['label']).cuda(), Variable(data['inst']).cuda(),
                                                   Variable(data['image']).cuda(), Variable(data['feat']).cuda())
                batch_loss_t.append(loss.item())
            epoch_loss_t.append(sum(batch_loss_t) / len(batch_loss_t))
        torch.cuda.empty_cache()
        return self.model.state_dict(), self.model.netG.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(
            epoch_loss_t) / len(epoch_loss_t)
