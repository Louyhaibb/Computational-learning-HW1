import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math

def load_data(file_path='kleibers_law_data.csv'):
    return pd.read_csv(file_path)

def plot_data(X, y, x_label, y_label, title, y_pred=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', marker='o', label='Observed data')
    if y_pred is not None:
        plt.plot(X, y_pred, color='red', label='Regression line')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_log_regression(X, y):
    log_X = np.log10(X)
    log_y = np.log10(y)
    model = LinearRegression()
    model.fit(log_X, log_y)
    y_pred = model.predict(log_X)
    return model.intercept_[0], model.coef_[0][0], y_pred

def metabolic_rate_prediction(mass, intercept, slope):
    log_mass = np.log10(mass)
    return 10 ** (intercept + slope * log_mass)

def energy_conversion(joules, to_calories=True):
    if to_calories:
        return joules / 4.18
    return joules * 4.18

if __name__ == "__main__":
    df = load_data()
    X = df[['mass']].values
    y = df[['metabolic_rate']].values

    intercept, slope, y_pred = perform_log_regression(X, y)
    print(f"Regression parameters - Intercept: {intercept:.3f}, Slope: {slope:.3f}")


    plot_data(np.log10(X), np.log10(y), 'Log of Mass (kg)', 'Log of Metabolic Rate (J/day)', 'Log-Log Linear Regression', y_pred)


    mass_example = 250
    metabolic_rate_example = metabolic_rate_prediction(mass_example, intercept, slope)
    calories_example = energy_conversion(metabolic_rate_example)
    print(f"Metabolic rate for {mass_example} kg mammal: {metabolic_rate_example:.2f} Joules/day ({calories_example:.2f} Calories/day)")


    metabolic_rate_target = 3500
    log_metabolic_rate = np.log10(metabolic_rate_target)
    estimated_mass = 10 ** ((log_metabolic_rate - intercept) / slope)
    print(f"Estimated mass for a metabolic rate of {metabolic_rate_target} Joules/day: {estimated_mass:.2f} kg")
