import math
import random
from quicksort import quicksort
from mergesort import mergesort
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import time


random.seed(10)

def test_quick(l, mid_pivot=True):
	result = 0	
	for x in range(10):
		arr = copy(l)
		t = time.time()
		quicksort(l, mid_pivot=mid_pivot)
		result += time.time() - t
	return result

def test_merge(l):
	result = 0
	for x in range(10):
		arr = copy(l)
		t = time.time()
		mergesort(l)
		result += time.time() - t
	return result


def test_all(l):
	return test_quick(l, False), test_quick(l, True), test_merge(l)


def main():
	""" Produce a graph comparing the speed of my mergesort and quicksort algorithms for
	increasingly sized already sorted lists.
	
	The result is that quicksort with mid-range pivot selection is quicker than mergesort by a constant factor.
	However, quicksort with first-element pivot selection is a worst case scenario, with time complexity O(N^2).
	
	The reason is that each partition of the list will only ever be one element smaller than the last, so there
	will be N partitions, and each partition operation takes O(N) time. Other options are:

	- Random pivot: Renders the worst-case scenario extremely unlikely.
	- Pivot at middle of the range (used here): a simple and for pre-sorted arrays, but worst-case sitll possible.
	- Pivot at the median: The median is the optimal pivot, but it takes too long to recalculate it every step.
	- Pivot at the median of the first/middle/last values: approximates the true median. Faster on average than
			a random pivot, but an array can still be crafted to trigger the worst case.
	"""
	quick_times = []
	quick_mid_times = []
	merge_times = []
	sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 100, 150, 200]
	for arr_len in sizes:
		l = list(range(arr_len))
		quick_t_start, quick_t_mid, merge_t = test_all(l)
		quick_times.append(quick_t_start)
		quick_mid_times.append(quick_t_mid)
		merge_times.append(merge_t)
	plt.plot(sizes, np.array([merge_times, quick_mid_times, quick_times]).T)
	plt.legend(["Mergesort", "Quicksort (Midpoint pivot)", "Quicksort (tart"], loc=2)
	plt.show()

if __name__ == '__main__':
	main()
