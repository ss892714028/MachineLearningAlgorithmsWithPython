# Hierarchical cluster's complexity is O(n^3 * m)
# runs extremely slow
# This implementation is only for educational purpose, speed is poorly optimized
# if you don't want your pc to catch on fire, run with a few hundred samples.
import numpy as np
import copy
import Data as d


class HierarchicalCluster:
    def __init__(self, test, num_cluster):
        self.test = np.array(test)[0:1000]
        self.num_cluster = num_cluster
        self.feature_space = self.test.shape[1]

    def calculate_distances(self, data):
        sample_size = len(data)
        distance_matrix = np.empty(shape=[sample_size,sample_size])
        for i in range(sample_size):
            if len(np.array(data[i]).shape) != 1:
                data[i] = list(np.array(data[i]).mean(axis=0))
        print(np.array(data).shape)
        for i in range(sample_size):
            for j in range(sample_size):
                distance_matrix[i, j] = self.distance(data[i], data[j])

        return distance_matrix

    @staticmethod
    def distance(x1, x2):
        # L2 distance
        return np.linalg.norm(np.array(x1) - np.array(x2))

    def cluster(self):
        data = self.test
        data = list(data)
        cluster = len(data)
        while cluster > self.num_cluster:
            print(cluster)
            distance_matrix = self.calculate_distances(data)
            d = distance_matrix.flatten()
            min_distance = min(d[np.nonzero(d)])
            coordinates = [np.where(distance_matrix == min_distance)]
            pairs = np.asarray(coordinates).T
            s = []
            for i in pairs:
                if sorted(i) not in s:
                    s.append(sorted(i))
                    data = self.merge(data, i)

            data = self.remove_by_indices(data, set(pairs.flatten()))
            cluster = len(data)
        return data

    @staticmethod
    def merge(lst, pair):
        temp = []
        for i in np.array(pair):
            temp.append(lst[int(i)])
        lst.append(temp)
        return lst

    @staticmethod
    def remove_by_indices(data, idxs):
        idxs = set(idxs)
        return [e for i, e in enumerate(data) if i not in idxs]


if __name__ == '__main__':
    test_data, test_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_test.csv')
    h = HierarchicalCluster(test_data, num_cluster=10)
    data = h.cluster()
    print(len(data))