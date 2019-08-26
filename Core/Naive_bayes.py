import numpy as np
import Data as d


class NaiveBayes:
    def __init__(self, train, test, train_label, test_label):
        self.train = np.array(train)
        self.test = np.array(test)
        self.train_label = train_label
        self.test_label = test_label
        self.c = list(set(self.test_label))

    def calculate_prob(self):
        data = self.train
        label = self.train_label
        # creates an empty dictionary, count occurance of each class
        counter = {}
        for i in label:
            if i in counter.keys():
                counter[i] += 1
            else:
                counter.update({i: 1})
        prob = []*0
        for index, value in enumerate(counter.keys()):
            prob[index] = counter[value] / len(label)






