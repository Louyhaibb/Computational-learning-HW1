import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression

class LR:

    def __init__(self, a1, a0, sample, num, stop, start, funci):
        self.a1 = a1
        self.a0 = a0
        self.sample = sample
        self.num = num
        self.stop = stop
        self.start = start
        self.funci = funci
        self.generate_data()
        self.fit_model()

    def generate_data(self):
        self.x = 10 * np.random.rand(self.sample)
        self.y = self.generate_function()

    def generate_function(self):
        if self.funci == 0:
            return self.a0 + self.a1 * self.x + np.random.randn(self.sample)
        else:
            return self.a1 + self.a0 * self.x + np.random.normal(0, 25, self.sample)

    def fit_model(self):
        self.model = LinearRegression()
        self.model.fit(self.x[:, np.newaxis], self.y)
        self.xfit = np.linspace(self.start, self.stop, self.num)
        self.yfit = self.model.predict(self.xfit[:, np.newaxis])

    def plot(self):
        plt.scatter(self.x, self.y, color='blue', label='Data points')
        plt.plot(self.xfit, self.yfit, color='red', label='Fit: y = {:.2f} + {:.2f}x'.format(self.model.intercept_, self.model.coef_[0]))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression Model')
        plt.legend()
        plt.show()

        print("Model slope: {:.2f}".format(self.model.coef_[0]))
        print("Model intercept: {:.2f}".format(self.model.intercept_))

# Example of usage
if __name__ == "__main__":
    model1 = LR(a1=1.8, a0=0.2, sample=100, start=0, stop=10, num=1000, funci=0)
    model1.plot()
    model2 = LR(a1=5, a0=2.7, sample=500, start=0, stop=35, num=1000, funci=1)
    model2.plot()
