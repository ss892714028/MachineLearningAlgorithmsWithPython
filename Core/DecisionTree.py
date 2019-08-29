import numpy as np
import Data as d
import time


class DecisionTree:
    def __init__(self, train, test, train_label, test_label, epsilon=0.1, continuous=True, bin=5):
        """

        :param train: training data
        :param test: testing data
        :param train_label: training label
        :param test_label: testing label
        :param continuous: if the data is continuous, needs to preprocess
        :param bin: if continuous, specify how many bins are desired
        """

        self.max_value = np.max(np.array(train).flatten())
        self.bin = bin
        if continuous:
            self.train = self.pre_process(np.array(train))
            self.test = self.pre_process(np.array(test))
        else:
            self.train = np.array(train)
            self.test = np.array(test)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)
        self.c = sorted(list(set(self.train_label)))
        self.epsilon = epsilon

    def pre_process(self, data):
        # Because Decision Tree is designed for discrete features,
        # if data is continuous, discretize it using pre_process method.
        dict = {}
        interval = self.max_value/self.bin
        # create a dictionary to store bin# and cutoff point
        for i in range(self.bin+1):
            dict[i] = i * interval
        d = []
        # iterate through every element in the dataset
        # same as for index in range(data.flatten())
        dimension = data.shape[0]
        for index, sample in enumerate(data):
            if index%1000 == 999:
                print('Pre-processing data sample {}: out of {}'.format(index, dimension))
            temp = []
            for value in sample:
                # iterate through each potential bin
                for i in dict.keys():
                    # check whether the element belongs to that bin
                    if dict[i] <= int(value) < (dict[i] + interval):
                        # if yes, return the bin number
                        temp.append(i)
            d.append(temp)
        return np.array(d)

    def calculate_entropy(self):
        train = self.train
        label = self.train_label
        H_dict = {}
        for i in range(label.shape[0]):
            if label[i] not in H_dict:
                H_dict[label[i]] = 1
            else:
                H_dict[label[i]] += 1
        dim = label.shape[0]
        prob = [0] * len(self.c)
        for index, value in enumerate(sorted(H_dict.keys())):
            prob[index] = H_dict[value] / dim
        clss_prob = [i/dim for i in H_dict.values()]
        H_D = self.get_H_D(clss_prob)

        H_D_A = []
        # iterate each feature, resulting (data.shape[0],)
        for feature in train.T:
            # create a set for that feature
            feature_set = set(feature)
            sum_of_H_D = []
            # formula on 'statistical learning method' page 75
            # iterate each potential value of that class
            for classes in feature_set:
                # select label where feature value == classes
                l = label[feature == classes]
                # create a set for l
                temp_set = set(l)
                p = []
                for i in temp_set:
                    # calculate p(Dik/Di)
                    # 'statistical learning method' page 74 formula 5.8
                    p.append(l[l == i].size / l.size)
                # store entropy for each classes(potential value of a feature)
                sum_of_H_D.append((feature[feature == classes].size/feature.size) * self.get_H_D(p))
            # sum total entropy for each feature
            H_D_A.append(np.sum(sum_of_H_D))
        return H_D, H_D_A

    @staticmethod
    def get_H_D(p):
        """

        :param p: a list of probabilities
        :return: entropy H(D)
        """
        return -np.sum([i * np.log2(i) if i != 0 else 0 for i in p])

    def calculate_information_gain(self):
        H_D, H_D_A = self.calculate_entropy()
        return [H_D-i for i in H_D_A]

    def find_max_gain(self):
        temp = self.calculate_information_gain()
        # return max information gain and max feature index
        return np.max(temp), temp.index(np.max(temp))

    @staticmethod
    def trim_data(data, label, index, value):
        new_data = []
        new_label = []
        for i in range(len(data)):
            if data[i][index] == value:
                new_data.append(data[i][0:index] + data[i][index + 1:])
                new_label.append(label[i])
        return new_data, new_label

    def tree(self):
        data = (self.train, self.train_label)

    def

if __name__ == '__main__':
    t = time.time()

    print('Loading data...')
    train_data, train_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_train.csv')
    test_data, test_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_test.csv')
    data = []
    for i in train_data:
        data.append([int(int(num) > 128) for num in i])

    classifier = DecisionTree(data,test_data,train_label,test_label,bin=2,continuous=False)
    classifier.calculate_entropy()
    print('time ultilized: {}'.format(time.time()-t))