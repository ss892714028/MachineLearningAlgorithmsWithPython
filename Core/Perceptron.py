import numpy as np
import Data as d
import time


class Perceptron:
    def __init__(self, train, test, train_label, test_label, lr=0.001, epoch=200):
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
        self.train_label = self.binary_label(np.array(train_label))
        self.test_label = self.binary_label(np.array(test_label))
        self.lr = lr
        self.epoch = epoch

    @staticmethod
    def binary_label(label):
        binary_label = []
        for i in label:
            if i == 0 :
                binary_label.append(0)
            else:
                binary_label.append(1)

        return binary_label

    def fit(self):
        train = self.train
        train_label = self.train_label
        feature_num, sample_num = train.shape[1], train.shape[0]
        # initialize weight size = feature size
        w = np.zeros([1,feature_num])
        b = 0
        # iterate number of epochs
        for i in range(self.epoch):
            print('epoch: {} out of {}'.format(i,self.epoch))
            # iterate training set
            for k in range(train.shape[0]):
                yi = train_label[k]
                xi = train[k]
                # if sample prediction is wrong, update weights
                if - yi * (np.dot(w, xi) + b) >= 0:
                    w = w + self.lr * yi * xi
                    b = b + self.lr * yi
            # shuffle training dataset
            np.random.shuffle(train)

        return w, b

    def testing(self):
        test, test_label = self.test, self.test_label
        w, b = self.fit()
        error = 0
        # iterate testing set
        for i in range(test.shape[0]):
            xi = test[i]
            yi = test_label[i]
            # if prediction is wrong
            if -yi * (np.dot(w, xi) + b) >= 0:
                error += 1
        accuracy = 1-error/len(test)
        print(accuracy)
        return accuracy


if __name__ == '__main__':
    train_data, train_label = d.loadData('../Data/mnist_train.csv')
    test_data, test_label = d.loadData('../Data/mnist_test.csv')
    t = time.time()
    classifier = Perceptron(train_data, test_data, train_label, test_label)
    classifier.testing()
    print('time ultilized: {}'.format(time.time() - t))
