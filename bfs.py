from graph import Grid, Point
from collections import deque

def BredthFirstSearch(g, start, target_value):
	# Bredth first search traverses a graph by starting at at root node
	# then checking all nodes one vertex away, then two, and so on.

	# This can be thought of searching "level by level" in a tree
	# or "along a growing frontier" on a grid.

	# The maing datastructures required are a FIFO queue and a map
	# The queue keeps a record of which nodes need to be checked next
	queue = deque()
	
	# The map stores the distance of each visisted node to the root.
	distances = {}

	# The first node to process is the start node, which is distance 0 from itself
	queue.append(start)
	distances[start] = 0

	# The algorithm runs until there is nothing left in the queue - then everything has been visited
	while len(queue) > 0:
		# The node is popped off the front of queue, and we get its distance
		current = queue.popleft()
		current_distance = distances[current]

		# In this version of the algorithm, we're searching for a specific value, so we check for it
		if g.value(current) == target_value:
			return current, current_distance

		# The nodes neighbours are added to the *back* of the queue, ensuring that they won't be processed
		# until other nodes at this level are processed.
		for neighbor in g.neighbors(current):
			# Make sure the node hasn't already been visisted by checking in the distances map.
			# We also check that it's value is nonnegative, since we represent 'walls' in the grid as -1.
			if neighbor not in distances and g.value(neighbor) >= 0:
				queue.append(neighbor)
				# The neighboring node is one step further away than this one
				distances[neighbor] = current_distance + 1

def main():
    # Try out the algorithm on a simple maze.
    themap = [
        [ 1, -1,  1,  1,  1, -1,  1,  1],
        [ 1, -1,  1, -1, 10, -1,  1,  1],
        [ 1,  1,  1, -1,  1,  1, -1,  1],
        [ 1,  1, -1, -1,  1, -1,  2,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1]]
    g = Grid(themap)
  
    position, distance = BredthFirstSearch(g, Point(0, 0), 10)
    print("Found '10' at {}, {} tiles from start.".format(position, distance))
  
if __name__ == '__main__':
    main()