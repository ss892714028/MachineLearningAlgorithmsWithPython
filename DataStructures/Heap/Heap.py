import math
import sys


class Heap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = [0] * (max_size + 1)
        self.heap[0] = sys.maxsize
        self.size = 0

    @staticmethod
    def get_left_index(index):
        return index * 2

    @staticmethod
    def get_right_index(index):
        return index * 2 + 1

    @staticmethod
    def get_parent(index):
        return math.floor((index - 1)/2)

    def swap(self, index_1, index_2):
        self.heap[index_1], self.heap[index_2] = self.heap[index_2], self.heap[index_1]

    def get_min(self):
        pass

    def extract_min(self):
        pass

    def insert(self, num):
        if self.size >= self.max_size:
            return None
        self.size += 1
        this_index = self.size
        self.heap[self.size] = num
        while self.heap[this_index] < self.heap[self.get_parent(this_index)]:
            self.swap(this_index, self.get_parent(this_index))
            this_index = self.get_parent(this_index)

    def is_leaf(self, index):
        if (index >= self.size//2) and (index <= self.size):
            return True
        else:
            return False

    def heapify(self):
        pass




