import numpy as np
import Data as d
import time


class NaiveBayes:
    def __init__(self, train, test, train_label, test_label, bin):

        self.max_value = np.max(np.array(train).flatten())
        self.bin = bin

        self.train = self.pre_process(np.array(train))
        self.test = self.pre_process(np.array(test))
        self.train_label = train_label
        self.test_label = test_label
        self.c = sorted(list(set(self.test_label)))
        self.l = 1

    # def pre_process(self, data):
    #     d = []
    #     for sample in data:
    #         d.append([0 if i < self.break_point else 1 for i in sample])
    #     return np.array(d)

    def pre_process(self, data):
        dict = {}
        interval = self.max_value/self.bin
        for i in range(self.bin+1):
            dict[i] = i * interval
        d = []
        print(interval)
        print(dict)

        for index, sample in enumerate(data):
            if index%1000 == 999:
                print(index)
            temp = []
            for value in sample:

                for i in dict.keys():
                    if dict[i] <= int(value) < (dict[i] + interval):
                        temp.append(i)
            d.append(temp)
        return np.array(d)

    def calculate_prob(self):
        data = self.train
        train_label = self.train_label
        # creates an empty dictionary, count occurance of each class
        counter = {}
        for i in train_label:
            if i in counter.keys():
                counter[i] += 1
            else:
                counter.update({i: 1})

        prob = [0] * len(self.c)

        # calculate probability of p(y=y)
        # with laplace smoothing
        # p(y=ck)=sum(I(yi=ck)+L) / (N+(K*L))

        for index, value in enumerate(sorted(counter.keys())):
            prob[index] = (counter[value] + self.l) / (len(train_label) + len(self.c) * self.l)
        prob = np.log(prob)
        # calculate conditional probability p(x=x|y=y)
        # first store all occurances of (label,feature_index,feature_value)
        total_prob = np.zeros([len(self.c), data.shape[1], self.bin+1])
        # iterate the whole dataset
        for index, sample in enumerate(data):
            label = train_label[index]
            # iterate elements in featurespace
            for j in range(data.shape[1]):
                total_prob[label][j][sample[j]] += 1

        # calculate conditional probabilities
        # iterate each class
        for i in range(len(self.c)):
            print('Calculating prob for class {}'.format(i))
            # iterate each feature
            for j in range(data.shape[1]):
                # iterate potential value of each feature
                pxy = []
                for k in range(len(total_prob[i][j])):
                    pxy.append(total_prob[i][j][k])

                for k in range(len(pxy)):
                    total_prob[i][j][k] = np.log(pxy[k] + self.l) / (sum(pxy) + len(self.c) * self.l)

        return prob, total_prob

    def model(self, py, px_y, x):
        all_prob = [0] * len(self.c)
        for i in range(len(self.c)):
            log_prob = 0
            for j in range(self.train.shape[1]):
                log_prob += px_y[i][j][x[j]]
            all_prob[i] = py[i] + log_prob

        return all_prob.index(max(all_prob))

    def t(self):
        test, test_label = self.test, self.test_label
        errors = 0
        print('Calculating Probability P(Y=Yk)...')
        prob, total_prob = self.calculate_prob()
        for i in range(len(test)):
            if i%1000 == 999:
                print('Predicting test number: {}'.format(i))
            prediction = self.model(prob, total_prob, test[i])
            if prediction != test_label[i]:
                errors += 1
        accuracy = 1-errors/len(test)

        print(accuracy)
        return accuracy


if __name__ == '__main__':
    t = time.time()

    print('Loading data...')
    train_data, train_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_train.csv')
    test_data, test_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_test.csv')

    classifier = NaiveBayes(train_data,test_data,train_label,test_label,bin=5)
    classifier.t()
    print('time ultilized: {}'.format(time.time()-t))











