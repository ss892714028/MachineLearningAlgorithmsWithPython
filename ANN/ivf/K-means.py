import numpy as np
import copy
import Data as d


class Kmeans:
    def __init__(self, test, k, max_iteration=100):
        self.test = np.array(test)
        self.k = k
        self.max_iteration = max_iteration

    @staticmethod
    def distance(x1, x2, axis=1):
        # L2 distance
        return np.linalg.norm(x1 - x2,axis = axis)

    def get_centroid(self):
        data = self.test
        # randomly pick k number of data as initial centroids
        indx = np.random.randint(data.shape[0], size=self.k)

        centroid = data[indx,:]
        return centroid

    def k_means(self):
        data = self.test
        # Randomly Initialize k Centroids
        centroid = self.get_centroid()
        cluster_assignment = np.zeros(data.shape[0])
        previous_centroid = np.zeros(centroid.shape)
        check = self.distance(centroid, previous_centroid, axis=None)

        # Stop when old centroid == new centroid
        # or reach maximum iteration
        iterator = 0
        while check != 0 and iterator < self.max_iteration:
            for i in range(data.shape[0]):
                # get distances between data[i] and each centroid
                distances = self.distance(data[i], centroid, axis=1)
                # record the previous centroid

                # select the closest centroid
                cluster_num = np.argmin(np.array(distances))
                # record data[i]'s closest centroid
                cluster_assignment[i] = cluster_num

            # record the previous centroid
            previous_centroid = copy.deepcopy(centroid)
            # iterate through every centroid
            for i in range(self.k):
                # for each cluster, recalculate centroid position based on
                # the mean value of data in that cluster

                centroid[i] = np.mean([data[j] for j in range(data.shape[0]) if cluster_assignment[j] == i], axis=0)

                # record the distance between new centroid and old centroid
            check = self.distance(centroid, previous_centroid, axis=None)
            iterator += 1
            print('current difference: {}'.format(check))

        result = []
        for i in range(self.k):
            result.append([data[j] for j in range(data.shape[0]) if cluster_assignment[j] == i])
        return result


if __name__ == '__main__':
    test_data, test_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_test.csv')
    k = Kmeans(test_data,k = 10)
    results = k.k_means()
    dict = {}
    for i in range(len(test_label)):
        dict[tuple(test_data[i])] = test_label[i]
    result = []
    for i in results:
        temp = []
        for j in i:
            temp.append(dict[tuple(j)])
        result.append(temp)
    print(np.array([sorted(i) for i in result]))
    for i in result:
        # print occurance of each class in each cluster

        print([list(i).count(j) for j in range(10)])

#############
# output
# [1, 1101, 126, 50, 27, 80, 37, 79, 67, 28]
# [728, 0, 13, 0, 1, 5, 20, 1, 7, 6]
# [13, 3, 19, 1, 17, 7, 651, 0, 6, 4]
# [6, 4, 31, 125, 1, 153, 2, 4, 632, 16]
# [177, 0, 24, 58, 29, 258, 224, 1, 73, 6]
# [48, 3, 44, 717, 0, 284, 4, 0, 110, 8]
# [3, 0, 18, 16, 436, 43, 10, 172, 19, 427]
# [3, 24, 735, 34, 0, 2, 5, 17, 11, 2]
# [0, 0, 5, 4, 471, 56, 5, 45, 40, 484]
# [1, 0, 17, 5, 0, 4, 0, 709, 9, 28]