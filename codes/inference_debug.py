import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import torch
from data_handler import GetAndProcessData
from models.inception_lstm_model import Inception_LSTM_model

# load model
# load train + dev data
# load scaler + features mask
# calc predictions for train + dev
# calc FP's + FN's for train + dev
# select 'most significant' (in dev) errors
# visualize 'bad prediction' features

class InferenceDebug:

    def __init__(self):

        self.data_path = '../mock_data/train_data/'
        self.objects_path = '../mock_data/data_objects'  # scaler + features mask
        self.experiment_path = "../trained_models/model=['Inception_LSTM']_adversarial=False_debug=False_2020-05-21_21-02-39"
        self.run_path = "/Run(model_name='Inception_LSTM', loss_name='BCE', optimizer_name='Adam', n_epochs=40, batch_size=4096, learning_rate=0.001, hidden_dim=20, n_layers=3, epsilon=0, weight_decay=0)"

    def fetch_data(self):
        gpd = GetAndProcessData()
        X_train, y_train, X_dev, y_dev = gpd.get_data(path=self.data_path, test_flag=False)
        scaler, active_features_mask = gpd.load_objects(self.objects_path)
        X_train = gpd.remove_bad_features(X_train, active_features_mask)
        X_dev = gpd.remove_bad_features(X_dev, active_features_mask)
        X_train = gpd.apply_scaler(X_train, scaler)
        X_dev = gpd.apply_scaler(X_dev, scaler)
        return X_train, X_dev, y_train, y_dev

    def get_model(self, n_features):
        # TODO save configs to each run in train
        # configs_path = self.experiment_path + '/experiment_configs.yaml'
        # with open(configs_path, 'r') as f:
        #     configs = yaml.load(f, Loader=yaml.FullLoader)
        # hidden_dim = configs['hidden_dim']
        # n_layers = configs['n_layers']

        # init the model class:
        label_dim = 1
        hidden_dim = 20
        n_layers = 3

        model = Inception_LSTM_model(n_features, hidden_dim, n_layers, label_dim)
        model_path = self.experiment_path + self.run_path + '/trained_model/best_dev_model.torch'
        model.load_state_dict(torch.load(model_path))
        model.eval()  # set BN / dropouts to evaluation mode
        return model

    @staticmethod
    def calc_loss(model, X, y):
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        prediction = model(X)
        loss_func = torch.nn.BCELoss()
        loss = loss_func(prediction, y)
        return prediction.detach().numpy(), loss.item()

    @staticmethod
    def calc_acc(y, prediction):
        pred_bool = np.round(prediction)
        acc = np.sum(pred_bool == y) / len(y)
        return acc

    @staticmethod
    def plot_errors(y, preds, text):
        pred_y0 = preds[y == 0]
        pred_y1 = preds[y == 1]
        bins = np.linspace(0, 1, 10)
        plt.hist(pred_y0, bins=bins, label='y=0', alpha=0.5)
        plt.hist(pred_y1, bins=bins, label='y=1', alpha=0.5)
        plt.legend()
        plt.xlabel('prediction values')
        plt.ylabel('count')
        plt.title(text)
        plt.grid()
        plt.show()

    @staticmethod
    def get_specific_obs_indices(y, preds):
        # use on dev first
        arr = np.concatenate(
            (
                np.arange(len(y)).reshape(-1, 1),
                y.reshape(-1, 1),
                preds.reshape(-1, 1)
            ),
            axis=1)

        # TP's:
        TPs_arr = arr[np.logical_and(y == 1, preds > 0.5), :]
        TP_row = np.argsort(-TPs_arr[:, 2])[0]  # same as argmax. replace 0 with: 1, 2, 3..
        best_TP_ind = TPs_arr[TP_row, 0]
        print(f'best_TP_ind = {best_TP_ind}, value = {TPs_arr[TP_row, 2]}')

        # TN's:
        TNs_arr = arr[np.logical_and(y == 0, preds <= 0.5), :]
        TN_row = np.argsort(TNs_arr[:, 2])[0]  # same as argmin. replace 0 with: 1, 2, 3..
        best_TN_ind = TNs_arr[TN_row, 0]
        print(f'best_TN_ind = {best_TN_ind}, value = {TNs_arr[TN_row, 2]}')

        # FP's:
        FPs_arr = arr[np.logical_and(y == 0, preds > 0.5), :]
        FP_row = np.argsort(-FPs_arr[:, 2])[1000]  # same as argmax. replace 0 with: 1, 2, 3..
        worst_FP_ind = FPs_arr[FP_row, 0]
        print(f'worst_FP_ind = {worst_FP_ind}, value = {FPs_arr[FP_row, 2]}')

        # FN's:
        FNs_arr = arr[np.logical_and(y == 1, preds <= 0.5), :]
        FN_row = np.argsort(FNs_arr[:, 2])[1000]  # same as argmin. replace 0 with: 1, 2, 3..
        worst_FN_ind = FNs_arr[FN_row, 0]
        print(f'worst_FN_ind = {worst_FN_ind}, value = {FNs_arr[FN_row, 2]}')

        return int(best_TP_ind), int(best_TN_ind), int(worst_FP_ind), int(worst_FN_ind)

    def plot_features(self, X_dev, TP_ind, TN_ind, FP_ind, FN_ind):
        rcParams.update({'figure.autolayout': True,
                         'figure.figsize': [15, 8.5],
                         'font.size': 17})

        fig, ax_arr = plt.subplots(nrows=2, ncols=2)
        ax_arr = ax_arr.flatten()

        # TP
        ax = ax_arr[0]
        ax.imshow(X_dev[TP_ind, :, :], vmin=-3, vmax=4)
        ax.set_title(f'TP example, row={TP_ind}')
        ax.set_xlabel('features')
        ax.set_ylabel('days')

        # TN
        ax = ax_arr[1]
        ax.imshow(X_dev[TN_ind, :, :], vmin=-3, vmax=4)
        ax.set_title(f'TN example, row={TN_ind}')
        ax.set_xlabel('features')
        ax.set_ylabel('days')

        # FP
        ax = ax_arr[2]
        ax.imshow(X_dev[FP_ind, :, :], vmin=-3, vmax=4)
        ax.set_title(f'FP example, row={FP_ind}')
        ax.set_xlabel('features')
        ax.set_ylabel('days')

        # FN
        ax = ax_arr[3]
        ax.imshow(X_dev[FN_ind, :, :], vmin=-3, vmax=4)
        ax.set_title(f'FN example, row={FN_ind}')
        ax.set_xlabel('features')
        ax.set_ylabel('days')

        cax = fig.add_axes([0.1, 0.485, 0.8, 0.01])
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cm.get_cmap('viridis'), norm=mpl.colors.Normalize(vmin=-3, vmax=4), orientation='horizontal')
        cbar.set_clim(-3.0, 4.0)

        # show all subplots:
        fig.tight_layout()
        plt.show()


    def run_all(self):
        X_train, X_dev, y_train, y_dev = self.fetch_data()
        n_feature = X_train.shape[2]
        model = self.get_model(n_feature)
        pred_train, loss_train = self.calc_loss(model, X_train, y_train)
        pred_dev, loss_dev = self.calc_loss(model, X_dev, y_dev)
        print(f'loss train = {loss_train}')
        print(f'loss dev = {loss_dev}')
        acc_train = self.calc_acc(y_train, pred_train)
        acc_dev = self.calc_acc(y_dev, pred_dev)
        print(f'acc train = {acc_train}')
        print(f'acc dev = {acc_dev}')
        self.plot_errors(y_train, pred_train, 'train')
        self.plot_errors(y_dev, pred_dev, 'dev')
        best_TP_ind, best_TN_ind, worst_FP_ind, worst_FN_ind = self.get_specific_obs_indices(y_dev, pred_dev)
        self.plot_features(X_dev, TP_ind=best_TP_ind, TN_ind=best_TN_ind, FP_ind=worst_FP_ind, FN_ind=worst_FN_ind)


inf = InferenceDebug()
inf.run_all()




