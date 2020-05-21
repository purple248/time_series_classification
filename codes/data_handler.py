import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer


class GetAndProcessData:


    def get_data(self, path, test_flag):

        with open(path + 'X_data', 'rb') as f: #.pickle
            X = pickle.load(f)  # X dims = [obs, ts, f]
        with open(path + 'y_data', 'rb') as f:
            y = pickle.load(f)

        # remove bad features 22, 57, 65 (negative time etc.):
        X = np.delete(X, [22, 57, 65], 2)


        # for debug:
        #X = np.float32(np.random.rand(4250, 28, 5))
        #y = np.float32(y)
        # X = X[0:1000, :, :]
        # y = y[0:1000, :]

        if test_flag:
            X_test = X
            y_test = y
            return X_test, np.squeeze(y_test)  # un-normalized

        else:
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.30, random_state=42)
            return X_train, np.squeeze(y_train), X_dev, np.squeeze(y_dev)  # un-normalized

    @staticmethod
    def add_const(X):
        X += 1

    def generate_data_for_scaler(self, X_train, objects_write_path):

        # use up to 10k obs for scaling:
        n_obs_for_sclaer = min(X_train.shape[0], 10000)
        random_sampling_mask = np.random.choice(np.arange(X_train.shape[0]), n_obs_for_sclaer, replace=False)

        # calc scaler only based n the last day in the sequence:
        active_features_mask = np.std(X_train[:, -1, :], axis=0) > 0#1.0  # or lower value - 0

        with open(objects_write_path + '/active_features_mask.pkl', 'wb') as file:
            pickle.dump(active_features_mask, file)

        # define the data for scalers calculation (last day in sequence):
        #data_for_scaler = np.squeeze(X_train.astype(np.float64)[:, -1, active_features_mask][random_sampling_mask])
        data_for_scaler = X_train.astype(np.float64)[:, -1, active_features_mask][random_sampling_mask]  #
        # add const to avoid zero-values of some features:
        self.add_const(data_for_scaler)

        return data_for_scaler, active_features_mask

    def calc_scaler(self, data_for_scaler, objects_write_path):

        # define scaler for the Gaussian similarity. It also returns zero-mean + unit variance!:
        scaler = PowerTransformer(method='box-cox', standardize=True, copy=True)  # method='yeo-johnson'
        scaler.fit(data_for_scaler)

        # save scaler:
        with open(objects_write_path + '/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        return scaler

    def remove_bad_features(self, X, active_features_mask):
        X = X[:, :, active_features_mask]
        return X

    def apply_scaler(self, X, scaler):
        self.add_const(X)
        # scale all days in the sequence:
        for i in range(X.shape[1]):
            X[:, i, :] = scaler.transform(np.squeeze(X[:, i, :]))
        return X

    @staticmethod
    def load_objects(objects_read_path):
        # load scaler and features mask:
        with open(objects_read_path + '/scaler.pkl', 'rb') as file:
            scaler0 = pickle.load(file)
        with open(objects_read_path + '/active_features_mask.pkl', 'rb') as file:
            active_features_mask = pickle.load(file)
        return scaler0, active_features_mask

    @staticmethod
    def do_clipping(X):
        # clip negative data (the scaler box-cox needs positive values)
        X = np.clip(X, a_min=0, a_max=None)
        return X

    def run_all_train(self, data_path, objects_write_path):
        X_train, y_train, X_dev, y_dev = self.get_data(path=data_path, test_flag=False)
        data_for_scaler, active_features_mask = self.generate_data_for_scaler(X_train, objects_write_path)
        scaler = self.calc_scaler(data_for_scaler, objects_write_path)
        X_train = self.remove_bad_features(X_train, active_features_mask)
        X_dev = self.remove_bad_features(X_dev, active_features_mask)
        X_train = self.apply_scaler(X_train, scaler)
        X_dev = self.apply_scaler(X_dev, scaler)
        return X_train, X_dev, y_train, y_dev  # scaled data

    def run_all_test(self, data_path, objects_read_path):
        X_test, y_test = self.get_data(path=data_path, test_flag=True)
        scaler, active_features_mask = self.load_objects(objects_read_path)
        X_test = self.remove_bad_features(X_test, active_features_mask)
        X_test = self.do_clipping(X_test)
        X_test = self.apply_scaler(X_test, scaler)
        return X_test, y_test  # scaled data



if __name__ == '__main__':
    print('start fetching data')
    data_train_path = '../mock_data/train_data/'
    data_test_path = '../mock_data/test_data/'
    objects_path = '../mock_data/data_objects'  # scaler and features mask
    gpd = GetAndProcessData()
    X_train, X_dev, y_train, y_dev = gpd.run_all_train(data_train_path, objects_path)
    X_test, y_test = gpd.run_all_test(data_test_path, objects_path) # use this with test data path
    print('data fetching successful')





