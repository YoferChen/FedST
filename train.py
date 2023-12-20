# import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd())))

from algorithm.local.local_trainer import LocalTrainer
from algorithm.fedavg.fedavg_trainer import FedAvgTrainer
from algorithm.fedprox.fedprox_trainer import FedProxTrainer
from algorithm.feddyn.feddyn_trainer import FedDynTrainer
from algorithm.feddc.feddc_trainer import FedDcTrainer
from data.data_processing.data_partion import get_partition_data_loader
import warnings
from util.util import Logger
import argparse
import time
import datetime
warnings.filterwarnings("ignore")

def load_data(args, dataset_name,log,opt_train,opt_test):
    train_data_num, test_data_num, train_data_global, val_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, val_data_local_dict \
        = get_partition_data_loader(args,log,opt_train,opt_test)


    dataset = [train_data_num, test_data_num, train_data_global, val_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, val_data_local_dict]

    return dataset

if __name__ == "__main__":
    start_time = time.time()
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    #get options
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'classification', choices=['material', 'medical', 'medical_material', 'face',  'aaf_face',], help='which dataset to use')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'fedst_ddpm'], help='chooses which model to use,unet...')
    parser.add_argument('--pretrained_model', type=str, help='weight file path of pretrained model...')
    parser.add_argument('--pretrained_model_for_DDPM', type=str, help='weight file path of DDPM pretrained model...')
    # fedddpm is fedst-separate
    parser.add_argument('--federated_algorithm', type=str, default='fedavg', choices=['local', 'fedavg', 'fedprox', 'feddyn', 'feddc', 'fedddpm', 'fedst_separate', 'fedst_join'], help='chooses which federated learning algorithm to use')
    parser.add_argument('--fake_dirname', default="", help='Fake image directory name for FedST algorithm')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids:e.g.0 0,1,2 1,2 use -1 for CPU')
    parser.add_argument('--client_id', type=int, default=-1, help='Client id for local trainer')
    op = parser.parse_args()

    if op.dataset == 'material':
        from opt.material_options import TrainOptions, TestOptions
    elif op.dataset == 'medical':
        from opt.medical_options import TrainOptions, TestOptions
    elif op.dataset == 'medical_material':
        from opt.medical_material_options import TrainOptions, TestOptions
    elif op.dataset == 'face':
        from opt.face_options import TrainOptions, TestOptions
    elif op.dataset == 'aaf_face':
        from opt.aaf_face_options import TrainOptions, TestOptions

    opt_train = TrainOptions().parse()
    opt_train.model = op.model
    opt_train.federated_algorithm = op.federated_algorithm
    opt_train.fake_dirname = op.fake_dirname
    opt_train.gpu_ids = op.gpu_ids
    opt_train.client_id = op.client_id
    opt_train.pretrained_model = op.pretrained_model
    opt_train.pretrained_model_for_DDPM = op.pretrained_model_for_DDPM
    opt_test = TestOptions().parse()
    opt_test.client_id = op.client_id
    print(opt_train.name)
    # opt_train.gpu_ids = [0]
    # set gpu ids
    gpu_ids = opt_train.gpu_ids.split(',')
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_ids])
    opt_train.gpu_ids = [i for i in range(len(gpu_ids))]
    # set cuda device
    if len(opt_train.gpu_ids) > 0:
        torch.cuda.set_device(opt_train.gpu_ids[0])

    # if opt_train.model == 'fedst_ddpm':
    #     opt_train.client_num_per_round = 20

    logging.disable(logging.DEBUG)
    log = Logger(opt_train.log_dir, level='debug')
    log.logger.info(opt_train)

    # load and partion dataset
    dataset = load_data(opt_train, opt_train.dataroot,log,opt_train,opt_test)

    if opt_train.federated_algorithm == 'local':
        trainer = LocalTrainer(dataset, opt_train, opt_test,log)
    elif opt_train.federated_algorithm == 'fedavg':
        trainer = FedAvgTrainer(dataset, opt_train, opt_test,log)
    elif opt_train.federated_algorithm == 'fedprox':
        trainer = FedProxTrainer(dataset, opt_train, opt_test,log)
    elif opt_train.federated_algorithm == 'feddyn':
        trainer = FedDynTrainer(dataset, opt_train, opt_test,log)
    elif opt_train.federated_algorithm == 'feddc':
        trainer = FedDcTrainer(dataset, opt_train, opt_test, log)
    elif opt_train.federated_algorithm == 'fedddpm':
        trainer = FedAvgTrainer(dataset, opt_train, opt_test, log)
    elif opt_train.federated_algorithm == 'fedst_separate':
        if not len(opt_train.fake_dirname):
            opt_train.fake_dirname = 'fake_image'
            opt_train.federated_algorithm = 'fedddpm'
        trainer = FedAvgTrainer(dataset, opt_train, opt_test, log)
    elif opt_train.federated_algorithm == 'fedst_join':
        if not len(opt_train.fake_dirname):
            opt_train.fake_dirname = 'fake_image_join'
            opt_train.federated_algorithm = 'fedddpm'
        trainer = FedAvgTrainer(dataset, opt_train, opt_test, log)
    else:
        raise Exception('algorithm not found!')

    trainer.train_cross_validation()
    end_time = time.time()
    total_time = end_time - start_time
    time = datetime.timedelta(seconds=total_time)
    print("Total Time: ", str(time))
    print(opt_train.model + '_' + opt_train.federated_algorithm + "Total Time: ", str(time))

