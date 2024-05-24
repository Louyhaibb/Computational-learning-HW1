import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class ReadData:
    def __init__(self, path='faithful.txt'):
        self.path = path
        self.data = np.loadtxt(self.path)

    def plot_data(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], color='red', marker='x')
        plt.xlabel('Duration of eruption (minutes)')
        plt.ylabel('Time to next eruption (minutes)')
        plt.title('Old Faithful Eruptions')
        plt.show()


class MiniBatchGradientDescent:
    def __init__(self, x, y, theta, alpha=0.01, minibatch_size=30, epochs=2000):
        self.x = np.hstack((np.ones((x.size, 1)), x.reshape(-1, 1)))
        self.y = y.reshape(-1, 1)
        self.theta = theta
        self.alpha = alpha
        self.minibatch_size = minibatch_size
        self.epochs = epochs

    def run(self):
        for _ in range(self.epochs):
            indices = np.random.permutation(self.x.shape[0])
            for idx in range(0, self.x.shape[0], self.minibatch_size):
                batch_indices = indices[idx:idx + self.minibatch_size]
                x_batch = self.x[batch_indices]
                y_batch = self.y[batch_indices]
                predictions = x_batch.dot(self.theta)
                errors = predictions - y_batch
                gradient = x_batch.T.dot(errors) / self.minibatch_size
                self.theta -= self.alpha * gradient
        return self.theta


class LinearRegressionPlotter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.model = LinearRegression()

    def fit_and_plot(self):
        self.model.fit(self.x.reshape(-1, 1), self.y)
        plt.scatter(self.x, self.y, color='red', marker='x')
        plt.xlabel('Duration of eruption (minutes)')
        plt.ylabel('Time to next eruption (minutes)')
        plt.title('Old Faithful Eruptions')
        x_fit = np.linspace(min(self.x), max(self.x), 1000)
        y_fit = self.model.predict(x_fit.reshape(-1, 1))
        plt.plot(x_fit, y_fit, color='blue')
        plt.show()


if __name__ == '__main__':
    data_reader = ReadData()
    x, y = data_reader.data[:, 0], data_reader.data[:, 1]
    data_reader.plot_data()

    theta_initial = np.array([[0.5], [10]])
    for alpha in [0.01, 0.001, 0.02, 0.002]:
        mbgd = MiniBatchGradientDescent(x, y, theta_initial.copy(), alpha, 30, 2000)
        theta_final = mbgd.run()
        print(f"Final theta for alpha={alpha}: {theta_final.flatten()}")

        plotter = LinearRegressionPlotter(x, y)
        plotter.fit_and_plot()
