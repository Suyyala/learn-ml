import torch
import matplotlib.pyplot as plt
from sklearn import model_selection
from torch.utils.data import random_split, TensorDataset


from  linear_regression import LinearRegressionSGD, LinearRegressionBatch, NaiveRegression


def load_data():
    # generate data
    X = torch.randn(100, 1)
    y = 2 * X + 3 + torch.randn(100, 1)
    return TensorDataset(X, y)

if __name__ == "__main__":
    # load data
    print("Loading data...")
    dataset = load_data()
    print(len(dataset))
    epochs = 100
    learning_rate = 0.01
    # split data
    print("Splitting data...")
    # Compute lengths for train and test
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    X_train, y_train = train_dataset[:]
    X_test, y_test = test_dataset[:]
    # train model
    model = LinearRegressionSGD(epochs, learning_rate)
    model.fit(X_train, y_train)
    # predict
    y_pred_test = model.predict(X_test)

    # plot
    X = dataset[:, 0]
    y = dataset[:, 1]
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color='red')
    plt.show()

    # print accuracy

    print(f"RMSE Error with SGD udpate {model.rmse_error(y_test, y_pred_test)}")

    # train with batch gradient descent
    model = LinearRegressionBatch(epochs, learning_rate)
    model.fit(X_train, y_train)
    # predict
    y_pred_test = model.predict(X_test)
    print(f"RMSE Error with BATCH udpate {model.rmse_error(y_test, y_pred_test)}")

    # train with naive regression
    model = NaiveRegression()
    model.fit(X_train, y_train)
    # predict
    y_pred_test = model.predict(X_test)
    print(f"RMSE Error with NAIVE model {model.rmse_error(y_test, y_pred_test)}")
