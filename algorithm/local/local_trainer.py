import numpy as np
import matplotlib.pyplot as plt
from algorithm.local.client import Client
import time, datetime
import torch
import os
from models import create_model
import copy


class LocalTrainer(object):
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

    def train_cross_validation(self):
        w_global_init = copy.deepcopy(self.model.state_dict())
        save_path = 'model_local'
        save_path += f'_client{self.opt_train.client_id}'

        timestamp = time.time()
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        formatted_string = dt_object.strftime('%m%d_%H%M%S')
        save_path = save_path + '_' + formatted_string
        print("Folder name of result:", save_path)

        for fold_idx in range(self.opt_train.folds):
            if self.opt_train.model == 'fedst_ddpm' and fold_idx > 0:
                break
            if fold_idx > 0:
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

                client_indexes = [self.opt_train.client_id]

                self.log.logger.info("client_indexes = " + str(client_indexes))

                for idx in client_indexes:
                    torch.cuda.empty_cache()

                    self.client.update_local_dataset(idx, self.train_data_local_dict[fold_idx][idx],
                                                     self.val_data_local_dict[fold_idx][idx],
                                                     self.train_data_local_num_dict[fold_idx][idx])
                    self.client.update_state_dict(w_global)
                    if self.opt_train.model == 'unet':
                        w, loss, loss_t = self.client.train(w_global, round_idx)
                    w_locals.append((self.client.get_sample_number(), copy.deepcopy(w)))
                    loss_locals.append(loss)
                    loss_locals_t.append(loss_t)
                    self.log.logger.info('Client {:3d}, loss {:.3f}, test loss{:.3f}'.format(idx, loss, loss_t))
                # update global weights
                w_global = self.aggregate(w_locals)

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
