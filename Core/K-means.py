import numpy as np
import copy
import Data as d


class Kmeans:
    def __init__(self, test, k, max_iteration=100):
        self.test = np.array(test)
        self.k = k
        self.max_iteration = max_iteration

    @staticmethod
    def distance(x1, x2):
        # L2 distance
        return np.sqrt(np.sum(np.square(x1 - x2)))

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
        cluster = np.zeros(data.shape[0])
        previous_centroid = np.zeros(centroid.shape)
        check = 1
        # Stop when old centroid == new centroid
        # or reach maximum iteration
        iterator = 0
        while check != 0 and iterator < self.max_iteration:
            for i in range(data.shape[0]):
                distances = []
                # get distances between data[i] and each centroid
                for j in range(centroid.shape[0]):
                    distances.append(self.distance(centroid[j], data[i]))
                # record the previous centroid
                previous_centroid = copy.deepcopy(centroid)
                # select the closest centroid
                cluster_num = np.argmin(np.array(distances))
                # record data[i]'s closest centroid
                cluster[i] = cluster_num
            # iterate through every centroid
            print([i in cluster for i in range(10)])
            for i in range(self.k):
                # for each cluster, recalculate centroid position based on
                # the mean value of data in that cluster
                temp = [data[j] for j in range(data.shape[0]) if cluster[j] == i]
                if not temp:
                    print('aaa')
                    centroid[i] = np.zeros(data.shape[1])

                else:
                    centroid[i] = np.mean(temp)

                # record the distance between new centroid and old centroid
            check = self.distance(centroid, previous_centroid)
            iterator += 1
            print('current difference: {}'.format(check))

        result = []
        for i in range(self.k):
            result.append([data[j] for j in range(data.shape[0]) if cluster[j] == i])
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
        print([list(i).count(j) for j in range(10)])

