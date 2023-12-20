from util.util import adjust_learning_rate
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

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def update_state_dict(self, state_dict):
        for params in self.model.parameters():
            params.requires_grad = True
        self.model.load_state_dict(state_dict)

    def train(self, w_global, round_idx, alpha, local_update_last, global_update_last, global_model_param, hist_i):
        losses = {}
        epoch_loss = []
        self.model.train()
        test_data = self.local_test_data
        self.lr = adjust_learning_rate(self.args.lr, round_idx, self.args)
        self.log.logger.info('lr : ' + str(self.lr))
        for epoch in range(self.args.epochs):
            batch_loss = []
            for i, data in tqdm.tqdm(enumerate(self.local_training_data)):
                self.model.set_input(data)
                self.model.set_learning_rate(self.lr)
                self.model.feddc_optimize_parameters(alpha, local_update_last, global_update_last, global_model_param,
                                                     hist_i)
                losses['train_loss'] = self.model.cal_loss()
                batch_loss.append(losses['train_loss'])
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # Freeze model
        for params in self.model.parameters():
            params.requires_grad = False
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
        return self.model, sum(epoch_loss) / len(epoch_loss), sum(epoch_loss_t) / len(epoch_loss_t)
