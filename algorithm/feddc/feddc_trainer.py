import numpy as np
import matplotlib.pyplot as plt
from algorithm.feddc.client import Client
import time, datetime
import torch
import os
from models import create_model
import copy
from util.util import get_mdl_params, set_client_from_params


class FedDcTrainer(object):
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
            client_indexes = range(client_num_per_round)
        self.log.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def train_cross_validation(self):
        alpha = 0.005
        act_prob = 1
        w_global_init = copy.deepcopy(self.model.state_dict())

        if self.opt_train.model == 'fedst':
            save_path = 'model_fedst_feddc'
        else:
            save_path = 'model_feddc'
        timestamp = time.time()
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        formatted_string = dt_object.strftime('%m%d_%H%M%S')
        save_path = save_path + '_' + formatted_string
        print("Folder name of result:", save_path)

        for fold_idx in range(self.opt_train.folds):
            if fold_idx > 0:
                break
            self.setup_clients()

            min_loss = 99999
            loss_train = []
            loss_test = []
            w_global = w_global_init

            self.model.load_state_dict(w_global_init)
            client_num = self.opt_train.client_num_in_total
            n_par = len(get_mdl_params([self.model])[0])
            state_gadient_diffs = np.zeros((client_num + 1, n_par)).astype('float32')  # including cloud state
            parameter_drifts = np.zeros((client_num, n_par)).astype('float32')
            cld_mdl_param = get_mdl_params([self.model], n_par)[0]
            init_par_list = get_mdl_params([self.model], n_par)[0]
            clnt_params_list = np.ones(client_num).astype('float32').reshape(-1, 1) \
                               * init_par_list.reshape(1, -1)  # n_clnt X n_par

            self.log.logger.info(
                "####################################FOLDS:" + str(fold_idx) + " : {}".format(fold_idx))

            for round_idx in range(self.opt_train.comm_round):
                inc_seed = 0
                while True:
                    np.random.seed(round_idx + inc_seed)
                    act_list = np.random.uniform(size=client_num)
                    act_clients = act_list <= act_prob
                    selected_clnts = np.sort(np.where(act_clients)[0])
                    inc_seed += 1
                    if len(selected_clnts) != 0:
                        break
                global_model_param = torch.tensor(cld_mdl_param, dtype=torch.float32, device="cuda")  # Theta
                self.log.logger.info("################Communication round : {}".format(round_idx))

                w_locals, loss_locals, loss_locals_t = [], [], []
                delta_g_sum = np.zeros(n_par)

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
                        local_update_last = state_gadient_diffs[client_num]  # delta theta_i
                        global_update_last = state_gadient_diffs[-1]  # delta theta
                        hist_i = torch.tensor(parameter_drifts[idx], dtype=torch.float32, device="cuda")  # h_i
                        w, loss, loss_t = self.client.train(w_global, round_idx, alpha, local_update_last,
                                                            global_update_last, global_model_param, hist_i)
                    else:
                        raise Exception(f'FedDC not support model named {self.opt_train.model}')
                    w_locals.append(w)
                    curr_model_par = get_mdl_params([w_locals[idx]], n_par)[0]
                    delta_param_curr = curr_model_par - cld_mdl_param
                    parameter_drifts[idx] += delta_param_curr
                    beta = 1 / (
                        np.ceil(len(self.train_data_local_dict[fold_idx][idx]) / self.opt_train.batch_size)).astype(
                        np.int64) / self.opt_train.lr

                    state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
                    delta_g_cur = (state_g - state_gadient_diffs[idx])
                    delta_g_sum += delta_g_cur
                    state_gadient_diffs[idx] = state_g
                    clnt_params_list[idx] = curr_model_par
                    loss_locals.append(loss)
                    loss_locals_t.append(loss_t)
                    self.log.logger.info('Client {:3d}, loss {:.3f}, test loss{:.3f}'.format(idx, loss, loss_t))
                # update global weights
                avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)
                delta_g_cur = 1 / client_num * delta_g_sum
                state_gadient_diffs[-1] += delta_g_cur

                cld_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0)
                cur_cld_model = set_client_from_params(self.model, cld_mdl_param)
                w_global = copy.deepcopy(dict(cur_cld_model.state_dict()))

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
