import numpy as np
import Data as d
import time


class DecisionTree:
    def __init__(self, train, test, train_label, test_label, epsilon=0.1, continuous=True, discretize = 'binary', bin=2):
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
            if discretize == 'multi_bin':
                self.train = self.multi_class(np.array(train))
                self.test = self.multi_class(np.array(test))
            if discretize == 'binary':
                self.train = self.binarize(train)
                self.test = self.binarize(test)
        else:
            self.train = np.array(train)
            self.test = np.array(test)
        self.train_label = np.array(train_label)
        self.test_label = np.array(test_label)
        self.c = sorted(list(set(self.train_label)))
        self.epsilon = epsilon

    def binarize(self, d):
        data = []
        for sample in d:
            data.append([int(int(num) > 1) for num in sample])
        return np.array(data)

    def multi_class(self, data):
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
            if index % 1000 == 999:
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

    def calculate_entropy(self, trainData, trainLabel):
        train = trainData
        label = trainLabel
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
                for value in temp_set:
                    # calculate p(Dik/Di)
                    # 'statistical learning method' page 74 formula 5.8
                    p.append(len(l[l == value]) / len(l))
                # store entropy for each classes(potential value of a feature)
                sum_of_H_D.append((len(feature[feature == classes])/len(feature)) * self.get_H_D(p))
            # sum total entropy for each feature
            H_D_A.append(np.sum(sum_of_H_D))
        return H_D, H_D_A

    @staticmethod
    def get_H_D(lst):
        """

        :param p: a list of probabilities
        :return: entropy H(D)
        """
        return np.sum([-1 * p * np.log2(p) for p in lst])

    def find_max_gain(self, trainData, trainLabel):
        H_D, H_D_A = self.calculate_entropy(trainData, trainLabel)
        # return max information gain and max feature index
        return H_D_A.index(np.min(H_D_A)), H_D - np.min(H_D_A)

    @staticmethod
    def trim_data(data, label, index, value):
        new_data = []
        new_label = []
        for i in range(len(data)):
            if data[i][index] == value:
                new_data.append(np.hstack([data[i][0:index], data[i][index+1:]]))
                new_label.append(label[i])
        return new_data, new_label

    @staticmethod
    def find_class(label):
        classDict = {}
        for i in label:
            if i in classDict.keys():
                classDict[i] += 1
            else:
                classDict[i] = 1
        max_class = max(classDict, key=classDict.get)

        return max_class

    # recursive tree
    def build_tree(self, *data):
        trainData = np.array(data[0][0])
        trainLabel = np.array(data[0][1])

        classDict = {i for i in trainLabel}

        if len(classDict) == 1:
            return trainLabel[0]
        if len(trainData) == 0:
            return self.find_class(trainLabel)
        print('build node', len(trainData[0]), len(trainLabel))
        A, e = self.find_max_gain(trainData, trainLabel)

        print('feature selected: {}'.format(A))
        print('information gain: {}'.format(e))

        if e < self.epsilon:
            return self.find_class(trainLabel)

        treeDict = {A: {}}
        # for each discrete value,
        for i in range(self.bin):
            treeDict[A][i] = self.build_tree(self.trim_data(trainData, trainLabel, A, i))
        return treeDict

    def predict(self, test, tree):
        test = list(test)
        while True:
            (key, value), = tree.items()
            if type(tree[key]).__name__ == 'dict':
                d = test[key]
                del test[key]
                tree = value[d]
                if type(tree).__name__ == 'int32':
                    return tree
            else:
                return value

    def acc(self, tree):
        err = 0
        for i in range(len(self.test)):
            if self.test_label[i] != self.predict(self.test[i], tree):
                err += 1
        return 1-err/len(self.test)


if __name__ == '__main__':
    t = time.time()

    print('Loading data...')
    train_data, train_label = d.loadData(r'..\Data\mnist_train.csv')
    test_data, test_label = d.loadData(r'..\Data\mnist_test.csv')
    classifier = DecisionTree(train_data, test_data, train_label, test_label, bin=4, continuous=True, discretize='multi_bin')

    tree = classifier.build_tree((classifier.train, classifier.train_label))
    acc = classifier.acc(tree)
    print(acc)
    print('time ultilized: {}'.format(time.time()-t))
