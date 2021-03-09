import random
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(
            '%r  %2.2f ms' % \
            (method.__name__, (te - ts) * 1000))
        return result
    return timed


class BubbleSort:
    def __init__(self, arr):
        self.arr = arr

    @timeit
    def bubble_sort(self):
        arr = self.arr
        swapped = True
        k = 0
        while swapped:
            swapped = False
            for i in range(len(arr) - 1 - k):
                if arr[i] > arr[i+1]:
                    arr[i], arr[i+1] = arr[i+1], arr[i]
                    swapped = True
            k += 1
        return arr


class SelectSort:
    def __init__(self, arr):
        self.arr = arr

    @timeit
    def select_sort(self):
        arr = self.arr
        for i in range(len(arr)):
            min_index = i
            for k in range(i + 1, len(arr)):
                if arr[k] < arr[min_index]:
                    arr[k], arr[min_index] = arr[min_index], arr[k]
        return arr


@timeit
def python_sort(arr):
    return sorted(arr)


class QuickSort:
    def __init__(self, arr):
        self.arr = arr

    @staticmethod
    def partition(arr, start, end):
        i = start - 1
        pivot = arr[end]
        for j in range(start, end):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[end], arr[i + 1] = arr[i + 1], arr[end]
        return i + 1

    def sort(self, arr, start, end):
        if start < end:
            pivot_position = self.partition(arr, start, end)
            self.sort(arr, start, pivot_position - 1)
            self.sort(arr, pivot_position + 1, end)

    @timeit
    def quick_sort(self):
        self.sort(self.arr,0,len(self.arr)-1)


if __name__ == '__main__':
    long_arr = [random.randint(0, 50000) for i in range(4000)]
    sort_bubble = BubbleSort(long_arr)
    sort_bubble.bubble_sort()
    sort_select = SelectSort(long_arr)
    sort_select.select_sort()
    python_sort(long_arr)
    sort_quick = QuickSort(long_arr)
    sort_quick.quick_sort()