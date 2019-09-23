# a simple linear regression with least square estimator
# used boston dataset from sklearn.dataset
import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class LinearRegression:
    def __init__(self, train, test, train_label, test_label):
        """

        :param train: training data
        :param test: testing data
        :param train_label: training label
        :param test_label: testing label
        :param step_size: step size of the gradient ascend method
        :param iteration: number of iteration of the gradient ascend method
        """
        self.train = np.array(train)
        self.test = np.array(test)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)

    @staticmethod
    def add_feature(d):
        data = []
        for index, value in enumerate(d):
            data.append(np.append(d[index], 1))
        return np.array(data)

    def fit(self):
        x = self.train
        y = self.train_label
        x = self.add_feature(x)
        left = np.linalg.inv(np.dot(x.T, x))
        right = np.dot(x.T, y)
        result = np.dot(left, right)
        return result

    def predict(self):
        x = self.test
        x = self.add_feature(x)
        b = self.fit()

        y = np.dot(x, b)
        return y


if __name__ == '__main__':
    # load dataset
    # use same parameter as https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
    # to compare result with sklearn's implementation

    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
    Y = boston_dataset['target']
    # train test split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    # start timer after loading dataset
    t = time.time()
    classifier = LinearRegression(X_train, X_test, Y_train, Y_test)
    y_pred = classifier.predict()
    rmse = (np.sqrt(mean_squared_error(Y_test, y_pred)))
    print('time_ultilized: {}'.format(time.time()-t))
    print('rmse is: {}'.format(rmse))