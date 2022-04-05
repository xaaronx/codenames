import numpy as np


class Cosine:
    def __init__(self, array1: np.array, array2: np.array):
        self.array1 = array1
        self.array2 = array2

    def distance(self):
        if self.array1.ndim == 1:
            self.array1 = self.array1.reshape(1, self.array1.shape[0])
        sumyy = (self.array2 ** 2).sum(1)
        sumxx = (self.array1 ** 2).sum(1, keepdims=1)
        sumxy = self.array1.dot(self.array2.T)
        return (sumxy / np.sqrt(sumxx)) / np.sqrt(sumyy)


class DotProduct:
    def __init__(self, array1: np.array, array2: np.array):
        self.array1 = array1
        self.array2 = array2

    def distance(self):
        return np.dot(self.array1, self.array2.T)
