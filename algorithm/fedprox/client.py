import tqdm


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, model, log):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.log = log
        self.args = args
        self.model = model

    def update_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, round_idx):
        self.model.load_state_dict(w_global)
        losses = {}
        epoch_loss = []
        self.model.train()
        test_data = self.local_test_data
        self.lr = self.args.lr
        self.log.logger.info('lr : ' + str(self.lr))
        for epoch in range(self.args.epochs):
            batch_loss = []
            for data in tqdm.tqdm(self.local_training_data):
                self.model.set_input(data)
                self.model.set_learning_rate(self.lr)
                self.model.optimize_parameters()
                losses['train_loss'] = self.model.cal_loss()
                batch_loss.append(losses['train_loss'])
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

        return w, sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_t) / len(epoch_loss_t)
