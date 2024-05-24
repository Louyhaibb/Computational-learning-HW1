import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

sns.set()


class DataLoader:
    def __init__(self, *paths):
        self.paths = paths
        self.data = None

    def load_data(self):
        self.data = {path: np.load(path) for path in self.paths}
        return self.data



    def plot_data(self, x_label, y_label, title):
        for path in self.paths:
            data = np.load(path)
            if data.shape[1] < 2:
                print(f"Error: Data in {path} does not have enough columns.")
                continue
            plt.scatter(data[:, 0], data[:, 1], color='blue', marker='.')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.show()


class LinearRegressionModel:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.model = LinearRegression()

    def train_and_plot(self):
        self.model.fit(self.X.reshape(-1, 1), self.Y)
        Y_pred = self.model.predict(self.X.reshape(-1, 1))

        plt.scatter(self.X, self.Y, color='blue', marker='.')
        plt.plot(self.X, Y_pred, color='red', label='Linear Regression Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

        print("Linear Regression Model slope (a1) =", self.model.coef_[0])
        print("Linear Regression Model intercept (a0) =", self.model.intercept_)


class PolynomialRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.poly_features = PolynomialFeatures(degree=2)
        self.model = LinearRegression()

    def train_and_plot(self):
        X_poly = self.poly_features.fit_transform(self.X.reshape(-1, 1))
        self.model.fit(X_poly, self.y)

        X_range = np.linspace(self.X.min(), self.X.max(), 100).reshape(-1, 1)
        X_range_poly = self.poly_features.transform(X_range)
        plt.scatter(self.X, self.y, color='blue', label='Data')
        plt.plot(X_range, self.model.predict(X_range_poly), color='red', label='Polynomial Fit')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polynomial Regression of Second Order')
        plt.legend()
        plt.show()

        print("Polynomial Regression Coefficients:", self.model.coef_)
        print("Polynomial Regression Intercept:", self.model.intercept_)


if __name__ == '__main__':
    x_path = 'TA_Xhouses.npy'
    y_path = 'TA_yprice.npy'

    # Load data
    loader = DataLoader(x_path, y_path)
    data = loader.load_data()
    X, Y = data[x_path], data[y_path]
    loader.plot_data('Length of the house front (meters)', 'House price (NIS\'000s)','House Price Based on Front Length')


    print('Linear Regression:')
    lr_model = LinearRegressionModel(X, Y)
    lr_model.train_and_plot()

    print('\nPolynomial Regression of Second Order:')
    poly_model = PolynomialRegression(X, Y)
    poly_model.train_and_plot()
