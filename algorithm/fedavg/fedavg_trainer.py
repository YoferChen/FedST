import copy
import logging
import numpy as np
import matplotlib.pyplot as plt
from algorithm.fedavg.client import Client
import time, datetime
import torch
import os
from PIL import Image
from collections import OrderedDict
from models import create_model
import pdb
import copy
import random


class FedAvgTrainer(object):
    def __init__(self, dataset, opt_train, opt_test, log=None, gan=None):
        self.training_setup_seed(0)
        [train_data_num, test_data_num, train_data_global, val_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, val_data_local_dict] = dataset

        self.val_data_local_dict = val_data_local_dict
        self.model = create_model(opt_train)
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.client_list = []
        self.test_loss = []
        self.train_loss = []
        self.opt_train = opt_train
        self.opt_test = opt_test
        self.gan = gan
        self.client_mse = [[] for i in range(self.opt_train.client_num_in_total)]
        self.log = log

    def setup_clients(self):
        self.log.logger.info("############setup_clients (START)#############")
        self.client = Client(0, None, None, None, self.opt_train, self.model, self.log)
        self.log.logger.info("############setup_clients (END)#############")

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            total_indexes = list(range(client_num_in_total))
            client_indexes = random.sample(total_indexes, num_clients)  # 随机选择部分客户端参与联邦学习

        self.log.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def train_cross_validation(self):
        w_global_init = copy.deepcopy(self.model.state_dict())
        if self.opt_train.model == 'fedst_ddpm':
            save_path = 'model_fedavg_fedst_ddpm'
        else:
            save_path = 'model_fedavg'
        if self.opt_train.federated_algorithm == 'fedddpm':
            save_path = 'model_fedddpm'

        if self.opt_train.fake_dirname == 'fake_image':
            save_path = 'model_fedst_separate'
        elif self.opt_train.fake_dirname == 'fake_image_join':
            save_path = 'model_fedst_join'

        # get current timestamp
        timestamp = time.time()
        # timestamp to datetime
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        # datetime format
        formatted_string = dt_object.strftime('%m%d_%H%M%S')
        save_path = save_path + '_' + formatted_string
        print("Result will be saved to:", save_path)

        for fold_idx in range(self.opt_train.folds):
            if self.opt_train.model == 'fedst_ddpm' and fold_idx > 0:
                break  # run one fold to get ddpm
            if fold_idx > 0:  # run one fold for test
                break
            self.setup_clients()
            min_loss = 99999
            loss_train = []
            loss_test = []
            w_global = w_global_init
            self.log.logger.info(
                "####################################FOLDS:" + str(fold_idx) + " : {}".format(fold_idx))
            for round_idx in range(self.opt_train.comm_round):
                self.log.logger.info("################Communication round : {}".format(round_idx))

                w_locals, wg_locals, loss_locals, loss_locals_t = [], [], [], []

                client_indexes = self.client_sampling(round_idx, self.opt_train.client_num_in_total,
                                                      self.opt_train.client_num_per_round)

                self.log.logger.info("client_indexes = " + str(client_indexes))

                for idx in client_indexes:
                    torch.cuda.empty_cache()

                    self.client.update_local_dataset(idx, self.train_data_local_dict[fold_idx][idx],
                                                     self.val_data_local_dict[fold_idx][idx],
                                                     self.train_data_local_num_dict[fold_idx][idx])
                    self.client.update_state_dict(w_global)
                    # train on new dataset
                    if self.opt_train.model == 'unet':
                        w, loss, loss_t = self.client.train(w_global, round_idx)
                    elif self.opt_train.model == 'fedst_ddpm':
                        w, wg, loss, loss_t = self.client.train_ddpm(w_global, round_idx)
                    w_locals.append((self.client.get_sample_number(), copy.deepcopy(w)))
                    if self.opt_train.model == 'fedst_ddpm':
                        wg_locals.append((self.client.get_sample_number(), copy.deepcopy(wg)))
                    loss_locals.append(loss)
                    loss_locals_t.append(loss_t)
                    self.log.logger.info('Client {:3d}, loss {:.3f}, test loss{:.3f}'.format(idx, loss, loss_t))
                # update global weights
                w_global = self.aggregate(w_locals)
                if self.opt_train.model == 'fedst_ddpm':
                    wg_global = self.aggregate(wg_locals)

                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)
                loss_train.append(loss_avg)
                loss_avg_t = sum(loss_locals_t) / len(loss_locals_t)
                loss_test.append(loss_avg_t)
                self.log.logger.info(
                    'Round {:3d}, Average loss {:.3f}, Average test loss {:.3f}'.format(round_idx, loss_avg,
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
                if self.opt_train.model == 'fedst_ddpm':
                    torch.save(wg_global,
                               self.opt_train.dataroot + '/' + save_path + '/model' + str(round_idx) + '_folds' + str(
                                   fold_idx) + 'generator.pth')
                # Update training curve for each round
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

    def aggregate(self, w_locals):
        averaged_params = copy.deepcopy(w_locals[0][1])

        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num

        for k in averaged_params.keys():

            for i in range(0, len(w_locals)):
                local_model_params = w_locals[i][1][k]
                w = w_locals[i][0] / training_num
                if i == 0:
                    averaged_params[k] = w * local_model_params
                else:
                    averaged_params[k] += w * local_model_params
        return averaged_params

    def training_setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
