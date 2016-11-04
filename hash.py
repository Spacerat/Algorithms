from __future__ import print_function
import math


class HashMap:

    """ Similar to Python's 'dictionary', a hash map maps keys to objects.
    This is done by calculating a hash value from the object itself, and
    using it to decide the array index at which to store the object."""

    """ Provided there is little hash collision, the hash map has O(1)
    insertion and lookup time

    'Chaining' is used to deal with hash collision. The hash table
    actually stores linked-lists at each index. If two objects have
    the same hash, they will both be stored in the same list.
    """

    def __init__(self, N=32):
        """ Create a hash map, with an initial capacity N """
        self.size = N
        self.table = []
        self.count = 0

        # Initialise a list of lists
        for i in range(N):
            self.table.append([])

    def hasher(self, obj):
        """ Hash an object. (using its string representation). This uses a
        simple but well known hasing algorithm I found somewhere on the
        internet... """
        """ Normally in python one uses the __hash__ method of an object to
        get a suitable hash for it, but in this case I decided that was
        cheating """

        s = str(obj)
        value = 5381
        for c in s:
            value = ((value << 5) + value) + ord(c)
        return value

    def resize(self, newsize):
        """ Rebuild the hash table to have a new maximum capacity, by
        recalculating all the hashes of every object and moving everything
        to a new, larger array of lists. """
        oldtable = self.table
        self.table = []
        self.count = 0
        self.size = newsize

        # Initialise a new, larger table
        for i in range(newsize):
            self.table.append([])

        # Re-insert all key/value pairs from the old table
        for chain in oldtable:
            for k, v in chain:
                self.insert(k, v)

    def insert(self, key, value):
        """ Insert an object 'value' indexed by key 'key', or replace an
        existing object if the key is already in use """
        """ If the hash map's load exceeds 2/3 after this operation, the map
        is resized to double its capacity """

        if self.load > (2.0 / 3.0):
            self.resize(self.size << 1)

        # Look up the chain using the key's hash
        hashed = self.hasher(key) % self.size
        chain = self.table[hashed]

        # If the key already exists, replace it
        for i, v in enumerate(chain):
            if v[0] == key:
                chain[i] = (key, value)
                return

        # Otherwise, add it
        chain.append((key, value))
        self.count += 1

    def get(self, key):
        """ Get an item's value using its key, raising KeyError if the key
        is not present """
        hashed = self.hasher(key) % self.size
        chain = self.table[hashed]
        if len(chain) == 1:
            return chain[0][1]
        else:
            try:
                return (v for k, v in chain if k == key).next()
            except StopIteration:
                raise KeyError("'{}' not present".format(key))

    @property
    def load(self):
        """ Get the load of the hash map, i.e. the proportion of indicies
        occupied by objects. """
        return self.count / float(self.size)

    def remove(self, key):
        """ Remove the object stored with a specific key from the map """
        """ If the hash map's load is lower than 0.2 after this
        operation, it is resized to half its capacity """

        # Resize if needed
        if len(self) > 2 and self.load < 0.2:
            self.resize(self.size >> 1)

        # Look up the correct chain using the key's hash
        hashed = self.hasher(key) % self.size
        chain = self.table[hashed]

        # Delete the element in the list which has the specified key
        for i, kv in enumerate(chain):
            if kv[0] == key:
                del chain[i]
                self.count -= 1
                return

        # If the function has not yet returned, the key is not present
        raise KeyError("'{}' not present".format(key))

    def iter_pairs(self):
        """ Iterator over all stored (key, value) pairs """
        found = 0
        for chain in self.table:
            for i in chain:
                yield i
                found += 1
                if found == self.count:
                    return

    def __iter__(self):
        """ Iterator over all stored keys """
        return (k for k, v in self.iter_pairs())

    def keys(self):
        return list(iter(self))

    def iter_items(self):
        """ Iterator over all values """
        return (v for k, v in self.iter_pairs())

    def __str__(self):
        return "{{{}}}".format(
            ", ".join("{}: {}".format(k, v) for k, v in self.iter_pairs())
        )

    def __len__(self):
        """ Number of currently stored items """
        return self.count

    def __setitem__(self, key, value):
        self.insert(key, value)

    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        self.remove(key)


if __name__ == '__main__':
    m = HashMap()

    for x in range(100):
        m[x] = x * x

    print(m)

    for k in m.keys()[:]:
        del m[k]

    print(m.table)
    print(m.load)
