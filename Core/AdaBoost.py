import numpy as np
from sklearn.tree import DecisionTreeClassifier


# here we use sklearn decision tree implementation
# if we use my implementation, it will take way too much time.
class AdaBoost:
    def __init__(self, train, test, train_label, test_label, iterations,min_samples_split,
                 max_depth, ):
        self.train = train
        self.test = test
        self.train_label = train_label
        self.test_label = test_label
        self.iterations = iterations
        self.n = self.test_label.shape[0]
        self.weights = np.ones(self.n)/self.n
        self.trees = []
        for _ in range(iterations):
            self.trees.append(DecisionTreeClassifier(min_samples_split=min_samples_split,
                                                     max_depth=max_depth))

    @staticmethod
    def square_loss(y, y_hat):
        return 0.5 * np.power((y-y_hat), 2)

    def training_loop(self):
        y_pred = np.zeros(self.n)
        for i in range(self.iterations):







