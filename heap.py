from operator import ge, le
from math import floor
import random
class BinHeap(object):

	def __init__(self, items = None, maxheap=True):
		if items == None:
			items = []
		if maxheap:
			self.op = ge
		else:
			self.op = le
		self.array = list(items)
		self.push = self.insert

	def insert(self, key, val):
		arr = self.array
		index = len(arr)
		arr.append((key, val))
		self.up_heap(index)
	# def push(self, key, val):
	# 	self.insert(key, val)

	def empty(self):
		return len(self) == 0

	def pop(self):
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
		for i in xrange(int(floor((len(self))/2)), -1, -1):
			self.down_heap(i)
		
	def up_heap(self, index):
		arr = self.array
		parent_ind = int(floor((index-1) / 2))

		while index != 0:
			if self.op(arr[parent_ind][0], arr[index][0]):
				arr[index], arr[parent_ind] = arr[parent_ind], arr[index]
			else:
				break
			index = parent_ind
			parent_ind = int(floor((index-1) / 2))


	def down_heap(self, index):
		arr = self.array
		child_inds = lambda x: (x*2+1, x*2 + 2)
		c1, c2 = child_inds(index)

		while c1 < len(arr):
			if c2 >= len(arr):
				newind = c1
			else:
				newind = c2 if self.op(arr[c1][0], arr[c2][0]) else c1

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
	hp = BinHeap(((x, x) for x in items))
	while len(hp) > 0:
		yield hp.pop()[1]
heapsort = lambda x: list(heapsort_gen(x))


def main():
	hp = BinHeap(maxheap = False)
	hp.insert_many([(x, str(x)) for x in (random.randint(0, 10) for x in xrange(5))])
	hp.insert_many([(x, str(x)) for x in (random.randint(11, 20) for x in xrange(5))])
	for x in (random.randint(21, 30) for x in xrange(5)):
		hp.insert(x,str(x))

	print [hp.pop()[1] for x in xrange(len(hp))]
	print heapsort(["abc", "qrt", "aaa", "z", "aaaaa", "ljn"])

	hp = BinHeap()
	print hp.empty()
	hp.insert(4, 2)
	hp.insert(4, 3)
	print hp.empty()
	hp.pop()
	hp.pop()
	print hp.empty()

if __name__ == '__main__':
	main()