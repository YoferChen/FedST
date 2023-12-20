import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from algorithm.feddyn.client import Client
import torch
import os
from models import create_model
from PIL import Image
from collections import OrderedDict
import pdb
import copy
import time, datetime


class FedDynTrainer(object):
    def __init__(self, dataset, opt_train, opt_test, log=None, gan=None):
        self.training_setup_seed(0)
        [train_data_num, test_data_num, train_data_global, val_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, val_data_local_dict] = dataset
        self.test_global = test_data_global
        self.val_data_local_dict = val_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.model = create_model(opt_train)
        self.train_data_local_dict = train_data_local_dict
        self.client_list = []
        self.opt_train = opt_train
        self.opt_test = opt_test
        self.log = log
        self.h = {}

    def setup_clients(self):
        self.log.logger.info("############setup_clients (START)#############")
        self.client = Client(0, None, None, None, self.opt_train, self.model, self.log)
        if self.opt_train.cross_validation:
            for i in range(self.opt_train.folds):
                c_list = {}
                for client_idx in range(self.opt_train.client_num_per_round):
                    c_list[client_idx] = {}
                    local_gradient = {}
                    for k, v in self.model.named_parameters():
                        local_gradient[k] = torch.zeros_like(v).cpu()
                    c_list[client_idx]['local_gradient'] = local_gradient
                self.client_list.append(c_list)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            client_indexes = range(client_num_per_round)
        self.log.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def train_cross_validation(self):
        w_global_init = copy.deepcopy(self.model.state_dict())
        save_path = 'model_feddyn'

        timestamp = time.time()
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        formatted_string = dt_object.strftime('%m%d_%H%M%S')
        save_path = save_path + '_' + formatted_string
        print("Folder name of result:", save_path)

        self.model.load_state_dict(w_global_init)
        for fold_idx in range(self.opt_train.folds):
            if fold_idx > 0:
                break
            self.setup_clients()
            self.client.clean_optimizer_state()
            min_loss = 99999
            w_global = w_global_init
            loss_train = []
            loss_test = []
            self.setup_clients()
            # init h
            for k in self.model.state_dict():
                self.h[k] = torch.zeros_like(self.model.state_dict()[k])

            self.log.logger.info(
                "####################################FOLDS:" + str(fold_idx) + " : {}".format(fold_idx))

            for round_idx in range(self.opt_train.comm_round):
                self.log.logger.info("################Communication round : {}".format(round_idx))

                w_locals, loss_locals, loss_locals_t = [], [], []

                client_indexes = self.client_sampling(round_idx, self.opt_train.client_num_in_total,
                                                      self.opt_train.client_num_per_round)

                self.log.logger.info("client_indexes = " + str(client_indexes))
                for idx in client_indexes:
                    # update dataset
                    local_gradient = self.client_list[fold_idx][idx]['local_gradient']
                    self.client.update_local_dataset(idx, self.train_data_local_dict[fold_idx][idx],
                                                     self.val_data_local_dict[fold_idx][idx],
                                                     self.train_data_local_num_dict[fold_idx][idx],
                                                     local_gradient)

                    self.client.update_state_dict(w_global)

                    # train on new dataset
                    if self.opt_train.model == 'unet':
                        loss, loss_t, w, local_gradient = self.client.train(w_global, round_idx)
                    else:
                        raise Exception(f'FedDyn not support model named {self.opt_train.model}')
                    self.client_list[fold_idx][idx]['local_gradient'] = local_gradient

                    loss_locals.append(copy.deepcopy(loss))
                    loss_locals_t.append(copy.deepcopy(loss_t))
                    w_locals.append((self.client.get_sample_number(), copy.deepcopy(w)))
                    self.log.logger.info('Client {:3d}, loss {:.3f}, test loss{:.3f}'.format(idx, loss, loss_t))

                # update global weights
                w_global = self.aggregate(w_locals, w_global)

                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)
                loss_train.append(loss_avg)
                loss_avg_t = sum(loss_locals_t) / len(loss_locals_t)
                loss_test.append(loss_avg_t)
                self.log.logger.info(
                    'Rouwnd {:3d}, Average loss {:.3f}, Average test loss {:.3f}'.format(round_idx, loss_avg,
                                                                                         loss_avg_t))

                if loss_avg_t < min_loss:
                    if not os.path.exists(self.opt_train.dataroot + '/' + save_path + '/'):
                        os.mkdir(self.opt_train.dataroot + '/' + save_path + '/')
                    torch.save(w_global,
                               self.opt_train.dataroot + '/' + save_path + '/model' + str(round_idx) + '_folds' + str(
                                   fold_idx) + '_best.pkl')
                    min_loss = loss_avg_t
                torch.save(w_global,
                           self.opt_train.dataroot + '/' + save_path + '/model' + str(round_idx) + '_folds' + str(
                               fold_idx) + '.pkl')
                # 每轮通信实时更新训练曲线
                plt.figure()
                plt.plot(np.linspace(1, len(loss_train), len(loss_train)).astype(np.int), loss_train)
                plt.plot(np.linspace(1, len(loss_test), len(loss_test)).astype(np.int), loss_test)
                plt.legend(['train', 'test'])
                plt.savefig(self.opt_train.dataroot + '/' + save_path + '/a_temp_loss_' + str(fold_idx) + '.png')

            plt.figure()
            plt.plot(np.linspace(1, len(loss_train), len(loss_train)).astype(np.int), loss_train)
            plt.plot(np.linspace(1, len(loss_test), len(loss_test)).astype(np.int), loss_test)
            plt.legend(['train', 'test'])
            plt.savefig(self.opt_train.dataroot + '/' + save_path + '/loss_' + str(fold_idx) + '.png')

    def aggregate(self, w_locals, w_global):
        averaged_params = copy.deepcopy(w_locals[0][1])

        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num

        # update h
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i][1][k].to(w_global[k].data.device)
                w = w_locals[i][0] / training_num
                self.h[k] -= (self.opt_train.dyn_alpha * w * (local_model_params.data - w_global[k].data)).type(
                    self.h[k].type())

        # update global model
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i][1][k].to(w_global[k].data.device)
                w = w_locals[i][0] / training_num
                if i == 0:
                    averaged_params[k] = w * local_model_params
                else:
                    averaged_params[k] += w * local_model_params

            # not perform weighted average for parameters in bn module
            if not ('mean' in k or 'var' in k):
                averaged_params[k] -= 1 / self.opt_train.dyn_alpha * self.h[k]
        return averaged_params

    def training_setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
