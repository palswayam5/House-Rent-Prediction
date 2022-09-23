import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Linear_regression:
    def cost_function(X, y, theta):
        h = np.dot(X, theta)
        m = len(X)
        cost = np.sqrt((1/2*m)*sum(np.square(h-y)))
        return cost

    def gradient_descent(X, y, theta, alpha, no_of_iterations):
        m = len(X)
        h = np.dot(X, theta)
        for i in range(no_of_iterations):
            theta = theta - (alpha/m)*X.T.dot(h-y)
            print(Linear_regression.cost_function(X, y, theta))
        return theta

    def regularized_cost_function(X, y, theta, l):
        h = np.dot(X, theta)
        m = len(X)
        cost = (1/2*m)*sum(np.square(X.dot(theta)-y))
        cost = cost + (l/(2*m))*sum(theta**2)
        return cost

    def regularized_gradient_descent(X, y, theta, l, alpha, no_of_iterations):
        h = np.dot(X, theta)
        derivative_matrix = np.dot(X.T, (h-y))
        m = len(X)
        for i in range(no_of_iterations):
            theta[0] = theta[0] - (alpha/m)*derivative_matrix[0]
            theta[1:len(theta)] = theta[1:len(theta)] - (alpha/m) * \
                derivative_matrix[1:len(theta)] - \
                ((alpha*l)/m)*theta[1:len(theta)]
            print(Linear_regression.regularized_cost_function(X, y, theta, l))
        return theta

    def NormalEquation(X, y):
        print(Linear_regression.cost_function(
            X, y, np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)))
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


class Data_Cleaning:
    def category(df, column):
        return df[column].value_counts()

    def meanNormalization(df, column):
        df[column] = (df[column] - df[column].mean())/df[column].std()
        return df

    def one_hot_encode(df, column):
        encoded = pd.get_dummies(df[column], drop_first=True)
        df = df.drop(column, axis=1)
        df = pd.concat([df, encoded], axis=1)
        return df


class Data_Plotting:
    def correlation_plot(df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
