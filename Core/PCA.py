import numpy as np
import Data as d


class PCA:
    def __init__(self, data, critical_value):
        self.data = np.array(data)
        self.critical_value = critical_value

    def normalization(self):
        data = self.data
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        for i in range(data.shape[1]):
            if std[i] != 0:
                data[:, i] = (data[:, i] - mean[i]) / std[i]
            else:
                data[:, i] = 0
        return data, mean, std

    def find_eig(self):
        data, mean, std = self.normalization()
        cov_matrix = np.cov(data.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # sort the eigenvalues, return its sorted index
        order = np.absolute(eigenvalues).argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[order]

        return eigenvalues, eigenvectors

    def find_k(self, eigenvalues, eigenvectors):
        counter = 0
        while True:
            counter += 1
            if sum(eigenvalues[:counter])/sum(eigenvalues) > self.critical_value:
                break
        print('selected {} features'.format(counter))
        e_vector = eigenvectors.T[:counter]
        return e_vector

    def pca(self):
        data = self.data
        eigenvalues, eigenvectors = self.find_eig()
        e_vector = self.find_k(eigenvalues, eigenvectors)
        z = self.projection_data(data, e_vector)
        return z

    @staticmethod
    def projection_data(x, u):
        return np.dot(x, u.T)


if __name__ == '__main__':
    test_data, test_label = d.loadData(r'C:\Users\Stan\PycharmProjects\MachineLearningAlgorithms\Data\mnist_test.csv')
    reduce = PCA(test_data, 0.98)
    z = reduce.pca()
    print(z.shape)
    print(z[[1,2,3], :])
