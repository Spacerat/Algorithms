from __future__ import print_function

class LinkedList(object):
	def __init__(self):
		self.next = None
		self.last = self


	def __iter__(self):
		node = self.next
		while node:
			yield node.value
			node = node.next

	@property
	def tail(self):
	    return self.last
	

	def append(self, value):
		n = Node(self.last, value)
		self.last = n
		return n

	def delete(self, node):
		secondlast = node.prev
		node.delete()
		if self.last == node:
			self.last = secondlast

	def insert(self, node, value):
		n = Node(node, value)
		return n

	def __str__(self):
		return str([x for x in self])


class Node(object):
	def __init__(self, prev_node, value):
		self.value = value
		if prev_node:
			self.next = prev_node.next
			self.prev = prev_node
			if prev_node.next:
				prev_node.next.prev = self
			prev_node.next = self
		else:
			self.next = None
			self.prev = None

	def delete(self):
		self.prev.next = self.next
		if self.next:
			self.next.prev = self.prev
		self.prev = None
		self.next = None

	def __str__(self):
		return "Node({})".format(str(self.value))

l = LinkedList()

n1 = l.append("hello")
n2 = l.append("world")
n3 = l.append("!")

l.delete(n1)
l.delete(n3)

print(l)