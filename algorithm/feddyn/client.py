from collections import OrderedDict
import torch
from models.losses.losses import dice_loss
from util.util import adjust_learning_rate
import tqdm
import copy


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, model, log):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.log = log
        self.args = args
        self.model = model
        self.local_gradient = {}

    def update_state_dict(self, w_global):
        sd = self.model.state_dict()
        sd.update(w_global)
        self.model.load_state_dict(sd)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number,
                             local_gradient):
        self.local_gradient = local_gradient
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def clean_optimizer_state(self):
        try:
            self.model.optimizer_G.clean_state()
            self.model.optimizer_D.clean_state()
            self.model.optimizer_Seg.clean_state()
        except:
            pass

    def train(self, w_global, round_idx):
        self.model.load_state_dict(w_global)
        global_model = copy.deepcopy(self.model)
        losses = {}
        epoch_loss = []
        self.model.train()
        test_data = self.local_test_data
        self.lr = adjust_learning_rate(self.args.lr, round_idx, self.args)
        self.log.logger.info('lr : ' + str(self.lr))
        for epoch in range(self.args.epochs):
            epoch_iter = 0
            batch_loss = []

            for data in tqdm.tqdm(self.local_training_data):
                self.model.set_input(data)
                self.model.set_learning_rate(self.lr)
                self.model.set_requires_grad(self.model.net, True)
                self.model.optimize_parameters()
                losses['train_loss'] = self.model.cal_loss()
                lg = OrderedDict()
                for k in self.local_gradient.keys():
                    lg[k] = self.local_gradient[k].cuda()
                self.model.optimizers[0].step(local_gradient=lg)
                lg = None

                batch_loss.append(losses['train_loss'])

        for k, v in self.model.named_parameters():
            self.local_gradient[k] = self.local_gradient[k].cuda() - self.args.dyn_alpha * (
                    v.data - global_model.state_dict()[k].data)

        w = self.model.state_dict()
        self.model.eval()
        losses = {}
        epoch_loss_t = []
        batch_loss_t = []
        for data in tqdm.tqdm(test_data):
            data['label'][:, :, [0, -1], :] = 1
            data['label'][:, :, :, [0, -1]] = 1
            self.model.set_input(data)
            self.model.eval()
            self.model.test()
            losses['train_loss'] = self.model.cal_loss()
            batch_loss_t.append(losses['train_loss'])
        epoch_loss_t.append(sum(batch_loss_t) / len(batch_loss_t))
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_t) / len(epoch_loss_t), w, self.local_gradient
