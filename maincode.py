from Mymodel import DNNModel
from Mymodel import HyperparameterTuning
import Mymodel
import data
import numpy as np
import pandas as pd


def main():
    # import
    dataset = data.Dataset(pd.read_csv("../MNS_data_full.csv"))
    # get training and testing
    X_train: np.ndarray = dataset.training.input_matrix
    y_train: np.ndarray = dataset.training.out_matrix
    X_test: np.ndarray = dataset.testing.input_matrix
    y_test: np.ndarray = dataset.testing.out_matrix

    # DNNmodel
    train_input_shape = X_train.shape
    train_output_shape = y_train.shape

    # Hyperparameter adjustment
    learning_rates = [0.005, 0.001, 0.002, 0.003]
    epochs = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    batch_sizes = [4, 6, 8]
    # Instantiate the HyperparameterTuning class
    tuner = HyperparameterTuning(learning_rates, epochs, batch_sizes)

    # Tune hyperparameters: {"learning_rates", "epochs", "batch_sizes"}
    best_params, mean_mse_list, best_mse, pcc = (
        tuner.hyperparameters_tune(X_train, y_train,
                                   train_input_shape, train_output_shape,
                                   n_splits=10)
    )
    print(f"Best Hyperparameters: {best_params}", f"mean mse: {mean_mse_list}", f"best mse: {best_mse}", f"pcc: {pcc}")

    # get the best model

    best_model = DNNModel(train_input_shape, train_output_shape, best_params["learning_rate"])
    early_stopping = Mymodel.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    best_model.train(X_train, y_train,
                     epochs=best_params["epochs"],
                     batch_size=best_params["batch_size"],
                     early_stopping=early_stopping)
    mse, mean_pcc = best_model.evaluate(X_test, y_test)
    pred_matrix = best_model.predict(X_test)

    print(f"mse: {mse}", f"mean_pcc: {mean_pcc}")


if __name__ == "__main__":
    main()
