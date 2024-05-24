import numpy as np
import matplotlib.pyplot as plt



class ReadData:
    def __init__(self, path='Cricket.npz', name='arr_0'):
        self.path = path
        self.data = np.load(self.path)
        self.name = name
        self.x = self.data[self.name][:, 1]
        self.y = self.data[self.name][:, 0]

    def plot_data(self):
        plt.scatter(self.x, self.y)
        plt.xlabel('Temperature (ºF)')
        plt.ylabel('Chirps/Second')
        plt.title('Cricket Chirps vs Temperature')
        plt.grid(True)
        plt.show()

    def get_for_inference_for_gd(self):
        return self.x.reshape(-1, 1), self.y.reshape(-1, 1)


class GradientDescent:
    def __init__(self, x, y, theta0=0, theta1=0, iterations=100, learning_rate=0.01):
        self.x = np.hstack([np.ones((x.shape[0], 1)), x])
        self.y = y
        self.theta = np.array([theta0, theta1], dtype=float).reshape(-1, 1)
        self.iterations = iterations
        self.learning_rate = learning_rate

    def run(self):
        m = len(self.y)
        for _ in range(self.iterations):
            predictions = np.dot(self.x, self.theta)
            errors = predictions - self.y
            updates = self.learning_rate / m * np.dot(self.x.T, errors)
            self.theta -= updates
            yield self.theta, np.sum(errors ** 2) / (2 * m)

    def plot_regression_line(self):
        plt.scatter(self.x[:, 1], self.y, color='blue', label='Data points')
        plt.plot(self.x[:, 1], np.dot(self.x, self.theta), color='red', label='Regression Line')
        plt.xlabel('Temperature (ºF)')
        plt.ylabel('Chirps/Second')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    data_reader = ReadData()
    data_reader.plot_data()
    x, y = data_reader.get_for_inference_for_gd()
    gd = GradientDescent(x, y, theta0=2, theta1=0.5, iterations=1000, learning_rate=0.001)
    for theta, cost in gd.run():
        pass
    gd.plot_regression_line()
