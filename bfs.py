from graph import Grid, Point, pathify
from collections import deque

def XFirstSearch(g, start, target_value, depth_first=False):
    # Bredth and Depth first search both traverse graphs.
    # - Bredth first search traverses one 'level' at a time
    # - Depth first search goes deeper whenever possible.
    # The difference can be toggled with one tiny implementation change:
    # - Bredth first search puts the children of the current node at the back of the processing queue
    # - Depth first search puts the children of the current node at the front of the processing queue

    # The maing datastructures required are a queue and a map
    # The queue keeps a record of which nodes need to be checked next
    queue = deque()
    
    # The map stores the distance of each visisted node to the root.
    distances = {}
    parents = {}

    # The first node to process is the start node, which is distance 0 from itself
    queue.append(start)
    distances[start] = 0
    parents[start] = None

    # The algorithm runs until there is nothing left in the queue - then everything has been visited
    while len(queue) > 0:
        # The node is popped off the front of queue, and we get its distance
        current = queue.popleft()
        current_distance = distances[current]

        # In this version of the algorithm, we're searching for a specific value, so we check for it
        if g.value(current) == target_value:
            return current, current_distance, parents

        # The nodes neighbours are added to the *back* of the queue, ensuring that they won't be processed
        # until other nodes at this level are processed.
        for neighbor in g.neighbors(current):
            # Make sure the node hasn't already been visisted by checking in the distances map.
            # We also check that it's value is nonnegative, since we represent 'walls' in the grid as -1.
            if neighbor not in distances and g.value(neighbor) >= 0:
                # Put the neighbour at the front of the queue for depth first, or the back for bredth first.
                if depth_first:
                    queue.appendleft(neighbor)
                else:
                    queue.append(neighbor)
                # The neighboring node is one step further away than this one
                distances[neighbor] = current_distance + 1
                parents[neighbor] = current

def main():
    # Try out the algorithm on a simple maze.
    themap = [
        [ 1, -1,  1,  1,  1,  1,  1,  1],
        [ 1, -1,  1, -1, 10, -1,  1,  1],
        [ 1,  1,  1, -1,  1,  1, -1,  1],
        [ 1,  1,  1, -1,  1, -1,  2,  1],
        [ 1,  1,  1,  1,  1,  1,  1,  1]]
    g = Grid(themap)
    print("Bredth First Search\n-------------------")
    position, distance, parents = XFirstSearch(g, Point(0, 0), 10)
    print("Found '10' at {}, {} tiles from start.".format(position, distance))
    g.printWithPath(set(pathify(parents, position)))

    print("\nDepth First Search\n-------------------")
    position, distance, parents = XFirstSearch(g, Point(0, 0), 10, depth_first=True)
    print("Found '10' at {}, {} tiles from start.".format(position, distance))
    g.printWithPath(set(pathify(parents, position)))


    # Bredth First Search
    # -------------------
    # Found '10' at Point(row=1, col=4), 9 tiles from start.
    # ['**', '-1', '**', '**', '**', ' 1', ' 1', ' 1']
    # ['**', '-1', '**', '-1', '**', '-1', ' 1', ' 1']
    # ['**', '**', '**', '-1', ' 1', ' 1', '-1', ' 1']
    # [' 1', ' 1', ' 1', '-1', ' 1', '-1', ' 2', ' 1']
    # [' 1', ' 1', ' 1', ' 1', ' 1', ' 1', ' 1', ' 1']

    # Depth First Search
    # -------------------
    # Found '10' at Point(row=1, col=4), 19 tiles from start.
    # ['**', '-1', ' 1', ' 1', '**', '**', '**', ' 1']
    # ['**', '-1', ' 1', '-1', '**', '-1', '**', '**']
    # ['**', '**', '**', '-1', ' 1', ' 1', '-1', '**']
    # [' 1', ' 1', '**', '-1', ' 1', '-1', ' 2', '**']
    # [' 1', ' 1', '**', '**', '**', '**', '**', '**']
    # [Finished in 0.1s]
  
if __name__ == '__main__':
    main()
