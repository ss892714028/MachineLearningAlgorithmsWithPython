import numpy as np
import Data as d

class PCA:
    def __init__(self, data, k):
        self.data = np.array(data)
        self.k = k

