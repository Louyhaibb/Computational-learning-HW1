import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D


class ReadData:
    def __init__(self, path, delimiter):
        self.path = path
        self.delimiter = delimiter

    def load(self):
        return np.loadtxt(self.path, delimiter=self.delimiter)


def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


def prepare_data(data):
    X = np.hstack((np.ones((data.shape[0], 1)), data[:, :2]))
    y = data[:, 2].reshape(-1, 1)
    normalized_X, mean_X, std_X = normalize_features(X[:, 1:])
    return np.hstack((np.ones((data.shape[0], 1)), normalized_X)), y, mean_X, std_X


class LinearRegression3D:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = LinearRegression()

    def fit(self):
        self.model.fit(self.X, self.y)
        print("Model coefficients:", self.model.coef_)
        print("Model intercept:", self.model.intercept_)

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.scatter(self.X[:, 1], self.X[:, 2], self.y, color='blue', marker='o', label='Data Points')


        x1_range = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), num=20)
        x2_range = np.linspace(self.X[:, 2].min(), self.X[:, 2].max(), num=20)
        x1, x2 = np.meshgrid(x1_range, x2_range)
        y_pred = self.model.predict(
            np.hstack((np.ones((x1.size, 1)), x1.ravel().reshape(-1, 1), x2.ravel().reshape(-1, 1))))
        ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), color='red', alpha=0.5, label='Prediction Plane')

        ax.set_xlabel('Area')
        ax.set_ylabel('Rooms')
        ax.set_zlabel('Price')
        ax.set_title('3D Linear Regression')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    reader = ReadData('houses.txt', ',')
    data = reader.load()
    X, y, mean_X, std_X = prepare_data(data)


    lr_model = LinearRegression3D(X, y)
    lr_model.fit()
    lr_model.plot()
