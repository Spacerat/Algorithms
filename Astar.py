
from __future__ import print_function
from collections import namedtuple
from heap import BinHeap


class BaseGraph():
  """ A base definition of a graph which provides enough functionality to demonstrate A* """

  def neighbors(self, node):
    """ The most important feature a graph is the ability to get the neigbours of a node. """
    return []

  def cost(self, node1, node2):
    """ Get the traversal cost between two nodes"""
    return 1

  def heuristic(self, node1, node2):
    """ If possible, calculate a quick approximation/heuristic for total travel cost between two nodes. """
    return 1

Point = namedtuple('Point', ['row','col'])

class Grid(BaseGraph):
  """ A grid is one possible representation of a graph, where vertices are cartesian coordinates
  and each vertex has an edge to each neighbour."""
  def __init__(self, gridmap):
    self.grid = gridmap

  def neighbors(self, point):
    """ Return vertices one step away in each cardinal direction, ignoring points outside of the grid. """
    compass = ((point.row-1, point.col),
              (point.row+1, point.col),
              (point.row, point.col-1),
              (point.row, point.col+1))

    for nrowi, ncoli in compass:
      if nrowi < 0 or nrowi >= len(self.grid):
        continue
      if ncoli < 0 or ncoli >= len(self.grid[nrowi]):
        continue
      val = self.grid[nrowi][ncoli]
      if val >=0:
        yield Point(nrowi, ncoli)

  def cost(self, node1, node2):
    # Assume the nodes are adjacent. The movement cost is the value of the node being travelled to.
    # This results in directed cost, but that is OK.
    return 0 if node1 == node2 else self.grid[node2.row][node2.col]

  def heuristic(self, node1, node2):
    return abs(node2.row - node1.row) + abs(node2.col - node1.col)
        
def Astar(graph, start, goal, priority_with_cost=True, priority_with_heuristc=True):
  visited = {}
  parents = {}
  cost_to_start = {}

  # The 'open' set initially only contains the start point
  openset = BinHeap(maxheap = True)
  openset.push(0, start)
  parents[start] = None

  # The cost from the start vertex to the start vertex is 0.
  cost_to_start[start] = 0
  num_visits = 0

  while not openset.is_empty():

    # Consider the open vertex with the lowest cost-to-start/heuristic.
    currentval, current = openset.pop()

    # If we've found the goal, finish.
    if current == goal:
      break

    # Check each neighbour of this vertex
    for next_vertex in graph.neighbors(current):
      # Add the cost to this new vertex to the current path.
      new_cost = cost_to_start[current] + graph.cost(current, next_vertex)

      # If that cost is lower than the existing cost for that vertex, or that vertex has
      # not yet been visited...
      if next_vertex not in cost_to_start or new_cost < cost_to_start[next_vertex]:
        # Then update that vertex's cost-to-start with the new one.
        cost_to_start[next_vertex] = new_cost

        # Then add the vertex into the priority queue to be considered in the next iteration. 
        priority = new_cost if priority_with_cost else 0
        if priority_with_heuristc:
          priority += graph.heuristic(goal, next_vertex)
        openset.push(priority, next_vertex)

        # And record that the quickest way to this node is via the current vertex.
        parents[next_vertex] = current
        num_visits +=1

  print(num_visits, "vists")

  # By the end, the 'goal' vertex should point to a parent vertex which points to a parent,
  # and so on, going back to the start vertex. Following this chain gives us the algorithm's solution.
  return reversed(list(pathify(parents, goal)))


def pathify(parents, goal):
  # Traverse through a chain of parents from a goal value until there is no parent.
  current = goal
  while current:
    yield current
    current = parents[current]



def main():
  # Try out the algorithm on a simple maze.
  themap = [
    [ 1,  1,  1,  1,  1, -1,  1,  1],
    [ 1,  1,  1,  1, 10, -1,  1,  1],
    [ 1,  1,  1,  1,  1,  1, -1,  1],
    [ 1,  1, -1, -1, -1, -1,  2,  1],
    [ 1,  1,  1,  1,  1,  1,  1,  1]]
  g = Grid(themap)


  path = list(Astar(g, Point(0, 0), Point(0, 6)))
  newgrid = [['{:2}'.format(v) for v in l] for l in themap]
  
  for row, r in enumerate(newgrid):
    for col, c in enumerate(r):
      if Point(row, col) in path:
        newgrid[row][col] = "**"

    print(r)

if __name__ == '__main__':
  main()




"""
OPEN = priority queue containing START
CLOSED = empty set
while lowest rank in OPEN is not the GOAL:
  current = remove lowest rank item from OPEN
  add current to CLOSED
  for neighbors of current:
    cost = g(current) + movementcost(current, neighbor)
    if neighbor in OPEN and cost less than g(neighbor):
      remove neighbor from OPEN, because new path is better
    if neighbor in CLOSED and cost less than g(neighbor): **
      remove neighbor from CLOSED
    if neighbor not in OPEN and neighbor not in CLOSED:
      set g(neighbor) to cost
      add neighbor to OPEN
      set priority queue rank to g(neighbor) + h(neighbor)
      set neighbor's parent to current

reconstruct reverse path from goal to start
by following parent pointers
"""
