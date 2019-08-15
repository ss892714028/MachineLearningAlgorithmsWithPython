import numpy as np
import Data as d
import time

class KNN:
    def __init__(self, train, test, train_label, test_label, test_size = 200, k = 25):
        self.train = np.array(train)
        self.test = np.array(test)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)
        self.label_dim = len(set(test_label))
        self.k = k
        self.test_size = test_size

    def calculate_distance(self, x1, x2):
        return np.sqrt(np.sum(np.square(x1 - x2)))

    def get_closest(self, x):
        label = self.train_label
        distances = np.zeros(shape = self.train.shape[0])
        for index, value in enumerate(self.train):
            x1 = value
            dist = self.calculate_distance(x1=x1, x2=x)
            distances[index] = dist

        top_k_index = np.argsort(np.array(distances))[:self.k]

        label_list = [0] * self.label_dim

        for i in top_k_index:
            label_list[int(label[i])] +=1

        return label_list.index(max(label_list))

    def predict(self):
        errors = 0
        for i in range(self.test_size):
            print('test num: {} out of {}'.format(i, self.test_size))
            x = self.test[i]
            y = self.get_closest(x)
            if y != self.test_label[i]:
                errors += 1
        accuracy = 1 - (errors / self.test_size)
        print(accuracy)
        return accuracy


if __name__ == '__main__':
    t = time.time()

    train_data, train_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_train.csv')
    test_data, test_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_test.csv')

    classifier = KNN(train_data,test_data,train_label,test_label)
    classifier.predict()
    print('time ultilized: {}'.format(time.time()-t))