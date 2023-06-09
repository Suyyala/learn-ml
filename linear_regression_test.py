import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from  linear_regression import LinearRegressionSGD, LinearRegressionBatch, NaiveRegression


def load_data():
    # generate data
    X = np.random.randn(100, 1)
    y = 2 * X + 3 + np.random.randn(100, 1)
    return X, y

if __name__ == "__main__":
    # load data
    print("Loading data...")
    X, y = load_data()
    print(X.shape, y.shape)
    epochs = 100
    learning_rate = 0.01
    # split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    # train model
    model = LinearRegressionSGD(epochs, learning_rate)
    model.fit(X_train, y_train)
    # predict
    y_pred_test = model.predict(X_test)

    # plot
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
