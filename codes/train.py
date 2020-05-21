import yaml
import os
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from data_handler import GetAndProcessData
from collections import OrderedDict, namedtuple
from itertools import product
from models.inception_lstm_model import Inception_LSTM_model



class TrainingFlow:

    def __init__(self):

        self.data_path = '../mock_data/train_data/'
        self.objects_write_path = '../mock_data/data_objects'
        configs_path = '../configs/config_v1.yaml'

        self.configs = self.get_configs(configs_path)
        self.adversarial_flag = self.configs['adversarial_flag']
        self.debug_flag = self.configs['debug_flag']

        # training configurations search space:
        self.params = OrderedDict(
            model_name=self.configs['model_name'],
            loss_name=self.configs['loss_name'],
            optimizer_name=self.configs['optimizer_name'],
            n_epochs=self.configs['n_epochs'],
            batch_size=self.configs['batch_size'],
            learning_rate=self.configs['learning_rate'],
            hidden_dim=self.configs['hidden_dim'],
            n_layers=self.configs['n_layers'],
            epsilon=self.configs['epsilon'],
            weight_decay=self.configs['weight_decay']
        )

        self.label_dim = 1  # fixed variable

        # experiment name:
        self.EXP_NAME = f'model={self.params["model_name"]}_adversarial={self.adversarial_flag}_debug={self.debug_flag}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        print(self.EXP_NAME)


    # methods definition:

    @staticmethod
    def get_configs(configs_path):
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            return configs

    def save_configs(self, experiment_path):
        with open(experiment_path + '/experiment_configs.yaml', 'w') as f:
            yaml.dump(self.configs, f)

    def get_loss(self, loss_name):
        # returns the selected loss as a function
        if loss_name == 'Similarity':
            return self.similarity_loss
        elif loss_name == 'BCE':
            return torch.nn.BCELoss()

    @staticmethod
    def get_optimizer(optimizer_name):
        # returns the selected optimizer as a function
        if optimizer_name == 'Adam':
            return torch.optim.Adam
        elif optimizer_name == 'SGD':
            return torch.optim.SGD

    @staticmethod
    def get_runs(params):
        run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(run(*v))
        return runs

    def fetch_data(self, debug_flag):
        gpd = GetAndProcessData()
        X_train, X_dev, y_train, y_dev = gpd.run_all_train(self.data_path, self.objects_write_path)
        if debug_flag:
            return X_train[0:2000, :, :], X_dev, y_train[0:2000], y_dev
        else:
            return X_train, X_dev, y_train, y_dev

    @staticmethod
    def similarity_loss(pred, y, alpha):
        y = y.view(-1, 1)
        pred = pred.view(-1, 1)
        same_class = y == torch.t(y)
        pred_dist = torch.pow(pred - torch.t(pred), 2)
        L = pred_dist * same_class - pred_dist * (~same_class)
        L = torch.sum(L)
        # L = max(L, 0) + alpha
        return L

    @staticmethod
    def calc_acc(prediction, y):
        y_pred = np.round(prediction)
        acc = np.sum(y_pred == y) / len(y)
        return acc

    @staticmethod
    def forward_backward(optimizer, loss_func, model, X, y, include_backward_flag):
        X = torch.tensor(X, requires_grad=False)
        y = torch.tensor(y, requires_grad=False)
        pred = model(X)
        loss = loss_func(pred, y)
        if include_backward_flag:
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # back-propagation, compute gradients
            optimizer.step()  # move wights opposite to gradient
        return loss.item(), pred.detach().numpy()

    @staticmethod
    def get_correct_data_grad(X_correct, y_correct, model, optimizer, loss_func):
        X_correct = torch.tensor(X_correct, requires_grad=True)
        y_correct = torch.tensor(y_correct, requires_grad=False)

        for param in model.parameters():
            param.requires_grad = False

        pred_correct = model(X_correct)
        loss = loss_func(pred_correct, y_correct)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # back-propagation, compute gradients

        for param in model.parameters():
            param.requires_grad = True

        return X_correct.grad.data.numpy()

    @staticmethod
    def track_metrics(writer, train_metric, dev_metric, epoch, title):
        writer.add_scalars(title,
                           {
                               'train': train_metric,
                               'dev': dev_metric
                           },
                           epoch)

    def run_all(self):
        print('start')

        # read pre-processed data:
        X_train, X_dev, y_train, y_dev = self.fetch_data(debug_flag=self.debug_flag)

        # create an experiment directory:
        experiment_path = '../trained_models/' + self.EXP_NAME
        os.mkdir(experiment_path)
        self.save_configs(experiment_path)

        res_all_runs_dict = {}

        for run in self.get_runs(self.params):  # loop through the different params combinations

            res_run_dict = {'best epoch': None, 'loss train': None, 'loss dev': None, 'acc train': None, 'acc dev': None}

            print(run)  # current parameters combination

            run_path = experiment_path + '/' + str(run)
            os.mkdir(run_path)
            trained_model_path = run_path + '/trained_model'
            metrics_path = run_path + '/outputs'  # TensorBoard tracked metrics
            os.mkdir(trained_model_path)
            os.mkdir(metrics_path)

            # the model:
            n_feature = X_train.shape[2]
            model = Inception_LSTM_model(n_feature, run.hidden_dim, run.n_layers, self.label_dim)

            # get a loss function:
            loss_func = self.get_loss(run.loss_name)

            # init an optimizer instance:
            optimizer_func = self.get_optimizer(run.optimizer_name)
            optimizer = optimizer_func(model.parameters(), lr=run.learning_rate, weight_decay=run.weight_decay)

            with SummaryWriter(metrics_path) as writer:

                # for best model saving:
                dev_loss_best = 999999
                train_loss_best = 999999

                # define batches mask:
                I = np.arange(X_train.shape[0]) // run.batch_size
                for i_epoch in range(1, run.n_epochs+1):
                    model.train()
                    for i_batch in np.unique(I):
                        mask = I == i_batch

                        X_batch = X_train[mask, :, :]
                        y_batch = y_train[mask]

                        _, prediction_batch = self.forward_backward(optimizer, loss_func, model, X_batch, y_batch, include_backward_flag=True)

                        if self.adversarial_flag and run.epsilon > 0:
                            if i_epoch >= 5:  # start adversarial attack on this epoch
                                mask_correct = np.round(prediction_batch) == y_batch
                                X_correct = X_batch[mask_correct, :, :]
                                y_correct = y_batch[mask_correct]
                                X_correct_grad = self.get_correct_data_grad(X_correct, y_correct, model, optimizer, loss_func)
                                X_batch[mask_correct] = X_correct + run.epsilon * np.sign(X_correct_grad)
                                # update model:
                                self.forward_backward(optimizer, loss_func, model, X_batch, y_batch, include_backward_flag=True)

                    model.eval()

                    train_loss, prediction_train = self.forward_backward(optimizer, loss_func, model, X_train, y_train, include_backward_flag=False)
                    dev_loss, prediction_dev = self.forward_backward(optimizer, loss_func, model, X_dev, y_dev, include_backward_flag=False)

                    train_acc = self.calc_acc(prediction_train, y_train)
                    dev_acc = self.calc_acc(prediction_dev, y_dev)

                    print(f'Loss for epoch {i_epoch}: train = {train_loss}, dev = {dev_loss}')
                    print(f'Acc for epoch {i_epoch}: train = {train_acc}, dev = {dev_acc}\n')

                    np.random.shuffle(I)

                    self.track_metrics(writer, train_loss, dev_loss, i_epoch, 'Loss')
                    self.track_metrics(writer, train_acc, dev_acc, i_epoch, 'Accuracy')

                    # update best model:
                    if dev_loss < dev_loss_best:
                        dev_loss_best = dev_loss
                        best_epoch = i_epoch
                        # save best model:
                        torch.save(model.state_dict(), trained_model_path + '/best_dev_model.torch')
                        res_run_dict = {'best epoch': best_epoch, 'loss train': train_loss, 'loss dev': dev_loss, 'acc train': train_acc, 'acc dev': dev_acc}

                    # update best model:
                    if train_loss < train_loss_best:
                        train_loss_best = train_loss
                        # save best model:
                        torch.save(model.state_dict(), trained_model_path + f'/best_train_model.torch')

            res_all_runs_dict[run] = res_run_dict
            print(res_all_runs_dict)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    tfl = TrainingFlow()
    tfl.run_all()
    print('end')




