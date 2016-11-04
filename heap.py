from __future__ import print_function
from operator import ge, le
from math import floor
import random

class BinHeap(object):

	def __init__(self, items = None, maxheap=True):
		""" Set up a heap
		items: list of initial items, as (key/value) tuples 
		maxheap: bool, sort direction"""

		# A heap can be modeled as an array
		self.array = []
		if items == None:
			items = []
		if maxheap:
			self.op = ge
		else:
			self.op = le

		# Insert all initial items
		for item in items:
			self.insert(*item)
			print([k for k, v in self.array])


	def insert(self, key, val):
		""" Insert an value with a key """
		# Items are inserted at the bottom of the list, then moved up into place
		arr = self.array
		index = len(arr)
		arr.append((key, val))
		self.up_heap(index)

	def is_empty(self):
		return len(self) == 0

	def pop(self):
		""" Remove the root item and return it """
		# The root element is replaced with the last element, which is then moved
		# into place, resulting in a valid heap
		arr = self.array
		topval = self.array[0]
		arr[0] = arr[-1]
		arr.pop()
		self.down_heap(0)
		return topval

	def popval(self):
		return self.pop()[1]

	def insert_many(self, items):
		self.array.extend(list(items))
		self.heapify()


	def heapify(self):
		for i in range(int(floor((len(self))/2)), -1, -1):
			self.down_heap(i)
		
	def up_heap(self, index):
		""" Move an item up the heap until it is in a valid position """
		arr = self.array
		# This operation is worst case O(log N) since the number of levels is ceil(sqrt(N))

		# An item's parent is located halfway between itself and index 0.
		parent_ind = int(floor((index-1) / 2))

		while index != 0:
			# Swap up if the item is 'larger' than its parent (assuming maxheap)
			# Rr end otherwise
			if self.op(arr[parent_ind][0], arr[index][0]):
				arr[index], arr[parent_ind] = arr[parent_ind], arr[index]
			else:
				return
			# Find the new parent index another halfway up.
			index = parent_ind
			parent_ind = int(floor((index-1) / 2))


	def down_heap(self, index):
		""" Move an item in the heap down until it is in a valid position """
		arr = self.array
		# As with up_heap, this operation is O(log N).

		# An item's children are located twice as far down as the item itself
		child_inds = lambda x: (x*2+1, x*2 + 2)
		c1, c2 = child_inds(index)

		while c1 < len(arr):
			# Choose the 'largest' sibling (assuming maxheap)
			if c2 >= len(arr):
				newind = c1
			else:
				newind = c2 if self.op(arr[c1][0], arr[c2][0]) else c1

			# Swap the item with its largest sibling if it is smaller than the sibling
			# Or finish otherwise
			if self.op(arr[index], arr[newind]):
				arr[newind], arr[index] = arr[index], arr[newind]
				index = newind
				c1, c2 = child_inds(index)
			else:
				return

	def __len__(self):
		return len(self.array)

	def top(self):
		return self.array[0]

	def __str__(self):
		return str(self.array)

	def __repr__(self):
		return repr(self.array)


def heapsort_gen(items):
	""" Heapsort simply pops the heap until it is empty """
	# Since each pop operation is worst case O(logN) and N pops are required,
	# heapsort's worst case time is O(N logN)

	hp = BinHeap(((x, x) for x in items))
	while len(hp) > 0:
		yield hp.pop()[1]
heapsort = lambda x: list(heapsort_gen(x))


def main():
	hp = BinHeap(maxheap = False)
	hp.insert_many([(x, str(x)) for x in (random.randint(0, 10) for x in range(5))])
	hp.insert_many([(x, str(x)) for x in (random.randint(11, 20) for x in range(5))])
	for x in (random.randint(21, 30) for x in range(5)):
		hp.insert(x,str(x))

	print([hp.pop()[1] for x in range(len(hp))])
	print(heapsort(["abc", "qrt", "aaa", "z", "aaaaa", "ljn"]))

	hp = BinHeap()
	print(hp.is_empty())
	hp.insert(4, 2)
	hp.insert(4, 3)
	print(hp.is_empty())
	hp.pop()
	hp.pop()
	print(hp.is_empty())

if __name__ == '__main__':
	main()