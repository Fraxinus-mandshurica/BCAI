import os
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def safe_pearsonr(x, y):
    std_x = np.std(x)
    std_y = np.std(y)

    # Check if standard deviations are both non-zero
    if std_x > 0 and std_y > 0:
        return pearsonr(x, y)[0]
    else:
        # Handle constant input case
        return 0.0


class DNNModel:
    def __init__(self, inshape, outshape, learning_rate=0.001):
        """
        :param inshape: output should be mark * user * article, input is user * features
        :param outshape:
        :param learning_rate:
        """
        self.models = []
        self.model = None
        self.input_shape = inshape
        self.output_shape = outshape
        # model list for 7 marks separately
        for i in range(outshape[0]):
            model = self.build_model(learning_rate=learning_rate)
            self.models.append(model)

    def build_model(self, learning_rate):
        model = Sequential()
        model.add(Input(shape=(self.input_shape[1],)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.output_shape[2], activation='linear'))
        opt = Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32, early_stopping=None):
        for i, model in enumerate(self.models):
            # train model for each mark
            y_train_single_mark = y_train[i, :, :]
            callbacks = []
            if early_stopping:
                callbacks.append(early_stopping)

            model.fit(X_train, y_train_single_mark, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                      verbose=0)

    def predict(self, X_test):
        predictions = [model.predict(X_test) for model in self.models]
        return np.stack(predictions, axis=0)

    def evaluate(self, X_test, y_test: np.ndarray):
        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test.reshape(y_test.shape[0], -1),
                                 y_pred.reshape(y_pred.shape[0], -1))
        # Calculate PCC
        pcc_values = [safe_pearsonr(y_test[:, :, i].flatten(),
                                    y_pred[:, :, i].flatten()) for i in range(y_test.shape[2])]
        mean_pcc = np.mean(pcc_values)
        return mse, mean_pcc

    def cross_validation(self, X, y, epochs, n_splits, batch_size, early_stopping=None):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scores = []
        pcc_scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[:, train_index, :], y[:, test_index, :]

            self.train(X_train, y_train, epochs=epochs, batch_size=batch_size, early_stopping=early_stopping)

            mse, pcc = self.evaluate(X_test, y_test)
            mse_scores.append(mse)
            pcc_scores.append(pcc)

        return mse_scores, pcc_scores


class HyperparameterTuning:

    def __init__(self, learning_rates, epochs, batch_sizes):
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.batch_sizes = batch_sizes

    def hyperparameters_tune(self, X_train, y_train, input_shape, output_shape, n_splits=5):
        param_grid = list(ParameterGrid({'learning_rate': self.learning_rates,
                                         'epochs': self.epochs,
                                         'batch_size': self.batch_sizes
                                         }))

        best_params = None
        corr_pcc = None
        best_mse = float('inf')
        mean_mse_list = []
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        for params in param_grid:
            text_model = DNNModel(input_shape, output_shape, params['learning_rate'])
            mse_scores, pcc_scores = text_model.cross_validation(X_train, y_train,
                                                                 epochs=params['epochs'],
                                                                 n_splits=n_splits,
                                                                 batch_size=params['batch_size'],
                                                                 early_stopping=early_stopping)

            mean_mse = np.mean(mse_scores)
            mean_pcc = np.mean(pcc_scores)
            mean_mse_list.append(mean_mse)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_params = params
                corr_pcc = mean_pcc

        return best_params, mean_mse_list, best_mse, corr_pcc
