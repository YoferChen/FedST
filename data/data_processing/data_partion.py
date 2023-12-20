import numpy as np
import os
import pdb
import random
from data.dataloader.data_loader import get_dataloader
def get_partition_data_loader(args,log,opt_train,opt_test):
    X_train, y_train, net_train_dataidx_map= partition_data(args,log)
    #train_data_num = sum([len(net_train_dataidx_map[r]) for r in range(opt_train.client_number)])
    train_data_global, val_data_global, test_data_global = get_dataloader(opt_train,opt_test,log,if_test = True)
    log.logger.info("train_dl_global number = " + str(len(train_data_global)))
    log.logger.info("val_dl_global number = " + str(len(val_data_global)))
    log.logger.info("test_dl_global number = " + str(len(test_data_global)))

    #pdb.set_trace()

    # get local dataset
    X_train = list(X_train)
    y_train = list(y_train)
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    data_local_num_dict = dict()
    #pdb.set_trace()
    if args.cross_validation:
        train_data_num = 0
        val_data_num = len(val_data_global)
        net_dataidx_map = net_train_dataidx_map
        for i in range(args.folds):
            if i > 0:  # For Test
                break
            log.logger.info("##############################################folds: " +str(i)+'##################################')
            train_data_local_dict[i] = {}
            val_data_local_dict[i] = {}
            data_local_num_dict[i] = {}
            train_data_idx_dict = {}
            val_data_idx_dict = {}
            for client_idx in range(args.client_num_in_total):
                client = {}
                train_dataidxs = []
                #pdb.set_trace()

                for j in range(args.folds):
                    if j != i:
                        train_dataidxs =  train_dataidxs + net_dataidx_map[j][0][client_idx]
                local_data_num = len(train_dataidxs)
                data_local_num_dict[i][client_idx] = local_data_num
                val_dataidxs = net_dataidx_map[i][0][client_idx]
                data_name = [X_train, y_train, X_train, y_train]

                log.logger.info(f"Client: {client_idx}")
                train_data_local, val_data_local, _ = get_dataloader(opt_train,opt_test,log,train_dataidxs, val_dataidxs,data_name = data_name)
                train_data_local_dict[i][client_idx] = train_data_local
                val_data_local_dict[i][client_idx] = val_data_local
                train_data_idx_dict[client_idx] = train_dataidxs
                val_data_idx_dict[client_idx] = val_dataidxs


    return train_data_num, val_data_num, train_data_global, val_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, val_data_local_dict

def partition_data(args, log):
    log.logger.info("*********partition test***************")
    datadir = args.dataroot
    n_nets = args.client_num_in_total
    data_list = [ i for i in sorted(os.listdir(os.path.join(datadir , args.total_img_dir_real))) if not i.startswith('.')]
    label_list = [ i for i in sorted(os.listdir(os.path.join(datadir , args.total_label_dir_real))) if not i.startswith('.')]
    max_client = len(data_list)
    N = len([i for i in data_list if not i.startswith('.')])

    log.logger.info("N = " + str(N))

    length_data = len(data_list)
    index_real = np.linspace(0,length_data-1,length_data).astype(np.int)
    real_data_index = sorted(np.random.choice(index_real, length_data, replace=False))
    data_list = np.array(data_list)[real_data_index].tolist()
    label_list = np.array(label_list)[real_data_index].tolist()
    net_dataidx_map = [[] for _ in range(args.folds)]

    data_zip = list(zip(data_list, label_list))
    random.shuffle(data_zip)
    data_list, label_list = zip(*data_zip)

    index = np.arange(len(data_list))
    if args.data_type == 'mix' :
        idx_batch = [[] for _ in range(args.folds)]
        split = [i*int(len(data_list)/args.folds) for i in range(1,args.folds)]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(index,split))]

        #
        for i in range(args.folds):
            net_dataidx_map[i].append({})
            data_i = np.array(data_list)[idx_batch[i]]
            Austi = [ j for j in idx_batch[i] if np.array(data_list)[j].split('/')[-1].split('_')[0]=='Austenite']
            Iron = [j for j in idx_batch[i] if np.array(data_list)[j].split('/')[-1].split('_')[0]=='IronCrystal']
            log.logger.info(str(i) + ' fold : Austi :' + str(len(Austi)) + ' Iron :' + str(len(Iron)))
            nets_data = [Austi, Iron]

            for j in range(n_nets):
                net_dataidx_map[i][0][j] = nets_data[j]

    elif args.data_type == 'pure' :

        idx_batch = []
        data_nets_split = [int(i*len(data_list)/n_nets) for i in range(1,n_nets)]
        data_nets = [[] for _ in range(n_nets)]
        data_nets_index = [idx_j + idx.tolist() for idx_j, idx in zip(data_nets, np.split(index,data_nets_split))]
        #pdb.set_trace()
        for i in range(args.folds):
            net_dataidx_map[i].append({})
        for j in range(n_nets):
            #split = [i*int(len(data_nets_index[j])/args.folds) for i in range(1,args.folds)]
            split = [int(i*len(data_nets_index[j])/args.folds) for i in range(1,args.folds)]
            idx = [idx.tolist() for idx in np.split(data_nets_index[j],split)]
            for i in range(args.folds):
                net_dataidx_map[i][0][j] = idx[i]

    elif args.data_type == 'splited':
        data=[]
        label=[]
        for j in data_list:
            data += [ os.path.join(j,i) for i in sorted(os.listdir(os.path.join(datadir , args.total_img_dir_real,j))) if not i.startswith('.')]
            label += [ os.path.join(j,i) for i in sorted(os.listdir(os.path.join(datadir , args.total_label_dir_real,j))) if not i.startswith('.')]
        index = np.arange(len(data))
        idx_batch = []
        data_nets_split = [int(i*len(data)/n_nets) for i in range(1,n_nets)]
        data_nets = [[] for _ in range(n_nets)]
        data_nets_index = [idx_j + idx.tolist() for idx_j, idx in zip(data_nets, np.split(index,data_nets_split))]

        for i in range(args.folds):
            net_dataidx_map[i].append({})
        if n_nets==1:
            data_nets_split = [int(i*len(data)/max_client) for i in range(1,max_client)]
            data_nets = [[] for _ in range(max_client)]
            data_nets_index = [idx_j + idx.tolist() for idx_j, idx in zip(data_nets, np.split(index,data_nets_split))]
            for i in range(args.folds):
                net_dataidx_map[i][0][0] = []
            for j in range(max_client):
                split = [int(i*len(data_nets_index[j])/args.folds) for i in range(1,args.folds)]
                idx = [idx.tolist() for idx in np.split(data_nets_index[j],split)]

                for i in range(args.folds):
                    net_dataidx_map[i][0][0] += idx[i]

        else:
            for j in range(n_nets):
                split = [int(i*len(data_nets_index[j])/args.folds) for i in range(1,args.folds)]
                idx = [idx.tolist() for idx in np.split(data_nets_index[j],split)]
                for i in range(args.folds):
                    net_dataidx_map[i][0][j] = idx[i]

        data_list = data
        label_list = label

    net_train_dataidx_map = net_dataidx_map
    return data_list, label_list, net_train_dataidx_map
