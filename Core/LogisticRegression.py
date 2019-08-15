import numpy as np
import Data as d
import time


class LogisticRegression:
    def __init__(self, train, test, train_label, test_label, step_size = 0.001,
                 iteration = 200):
        self.train = np.array(train)
        self.test = np.array(test)
        self.train_label = self.binary_label(np.array(train_label))
        self.test_label = self.binary_label(np.array(test_label))
        self.step_size = step_size
        self.iteration = iteration

    @staticmethod
    def binary_label(label):
        binary_label = []
        for i in label:
            if i == 0 :
                binary_label.append(0)
            else:
                binary_label.append(1)

        return binary_label

    @staticmethod
    def add_feature(d):
        data = []
        for index, value in enumerate(d):
            data.append(np.append(d[index], 255)/255)
        return np.array(data)

    def get_weights(self):
        train = self.train
        label = self.train_label

        # add 1 feature space for b
        # y = wx+b
        train = self.add_feature(train)
        weights = np.zeros(train.shape[1])
        # epochs
        for i in range(self.iteration):
            print(i)
            # Iterate training set
            for j in range(train.shape[0]):
                y = label[j]
                x = train[j]
                wt = np.dot(weights, x)

                # gradient ascend
                weights += self.step_size * (x * y - ((np.exp(wt) * x) / (1 + np.exp(wt))))
        self.train = train

        return weights

    @staticmethod
    def predict(weights, x):
        product = np.dot(weights, x)
        probability_class1 = np.exp(product)/(1+np.exp(product))
        if probability_class1 >= 0.5:
            return 1
        else:
            return 0

    def t(self, weights):

        test, test_label = self.test, self.test_label
        test = self.add_feature(test)

        errors = 0
        for i in range(len(test)):
            prediction  = self.predict(weights, test[i])
            if prediction != test_label[i]:
                errors += 1
        accuracy = 1-errors/len(test)

        print(accuracy)
        return accuracy


if __name__ == '__main__':
    t = time.time()

    train_data, train_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_train.csv')
    test_data, test_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_test.csv')

    classifier = LogisticRegression(train_data,test_data,train_label,test_label)
    weights = classifier.get_weights()
    classifier.t(weights)
    print('time ultilized: {}'.format(time.time()-t))