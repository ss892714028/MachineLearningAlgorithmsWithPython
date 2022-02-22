import numpy as np
import copy


class Kmeans:
    def __init__(self, test, k, max_iteration=20):
        self.test = np.array(test)
        self.k = k
        self.max_iteration = max_iteration

    @staticmethod
    def distance(x1, x2, axis=1):
        # L2 distance
        return np.linalg.norm(x1 - x2, axis=axis)

    def get_centroid(self):
        data = self.test
        # randomly pick k number of data as initial centroids
        indx = np.random.randint(data.shape[0], size=self.k)

        centroid = data[indx, :]
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

                # centroid[i] = np.mean([data[j] for j in range(data.shape[0]) if cluster_assignment[j] == i], axis=0)
                centroid[i] = np.mean(data[np.where(cluster_assignment == i)], axis=0)
                # record the distance between new centroid and old centroid
            check = self.distance(centroid, previous_centroid, axis=None)
            iterator += 1

        result = []
        for i in range(self.k):
            result.append(data[np.where(cluster_assignment == i)])
        return result


if __name__ == '__main__':
    nlist = 100
    test_data = np.random.random([10000, 128])
    k = Kmeans(test_data, k=nlist)
    import time
    t = time.time()
    results = k.k_means()
    print(f"sida time spent: {time.time()-t}")

    from sklearn.cluster import KMeans
    import time
    t = time.time()
    kmeans = KMeans(n_clusters=nlist, max_iter=20).fit(test_data)
    print(f"sklearn time spent: {time.time() - t}")