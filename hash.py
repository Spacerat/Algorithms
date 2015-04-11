import math
class HashMap(object):
	def __init__(self, N=32):
		self.size = N
		self.table = []
		self.count = 0
		for i in xrange(N):
			self.table.append([])

	def hasher(self, obj):
		s = str(obj)
		value = 5381
		for c in s:
			value = ((value << 5) + value) + ord(c)
		return value

	def resize(self, newsize):
		oldtable = self.table
		self.table = []
		self.count = 0
		self.size = newsize
		for i in xrange(newsize):
			self.table.append([])
		for chain in oldtable:
			for k, v in chain:
				self.insert(k, v)



	def insert(self, key, value):
		if self.load > 0.666666666:
			self.resize(self.size << 1)
		hashed = self.hasher(key) % self.size
		chain = self.table[hashed]
		for i, v in enumerate(chain):
			if v[0] == key:
				chain[i] = (key, value)
				return
		chain.append((key, value))
		self.count += 1

	def get(self, key):
		hashed = self.hasher(key) % self.size
		chain = self.table[hashed]
		if len(chain) == 1:
			return chain[0][1]
		else:
			try:
				return (v for k, v in chain if k == key).next()
			except StopIteration as e:
				raise KeyError("'{}' not present".format(key))

	@property
	def load(self):
	    return self.count / float(self.size)

	def __len__(self):
		return self.count

	def remove(self, key):
		if len(self) > 2 and self.load < 0.2:
			self.resize(self.size >> 1)
		hashed = self.hasher(key) % self.size
		chain = self.table[hashed]
		for i, kv in enumerate(chain):
			if kv[0] == key:
				del chain[i]
				self.count -= 1
				return
		raise KeyError("'{}' not present".format(key))

	def items(self):
		found = 0
		for chain in self.table:
			for i in chain:
				yield i
				found += 1
				if found == self.count: return

	def iter_keys(self):
		return (k for k, v in self.items())
	def keys(self):
		return list(self.iter_keys())

	def values(self):
		return (v for k, v in self.items())


	def __setitem__(self, key, value):
		self.insert(key, value)

	def __getitem__(self, key):
		return self.get(key)

	def __delitem__(self, key):
		self.remove(key)

m = HashMap()

for x in xrange(100):
	m[x] = x*x

for k in m.keys()[:]:
	del m[k]

print m.table
print m.load



