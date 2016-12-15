from collections import namedtuple


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

  def value(self, node):
    """ Get a node's value data """
    return None

Point = namedtuple('Point', ['row','col'])

class Grid(BaseGraph):
  """ A grid is one possible representation of a graph, where vertices are cartesian coordinates
  and each vertex has an edge to each neighbour."""
  def __init__(self, gridmap):
    self.grid = gridmap

  def value(self, point):
    return self.grid[point.row][point.col]

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

  def printWithPath(self, path):
      newgrid = [['{:2}'.format(v) for v in l] for l in self.grid]
      
      for row, r in enumerate(newgrid):
        for col, c in enumerate(r):
          if Point(row, col) in path:
            newgrid[row][col] = "**"

        print(r)

def pathify(parents, goal):
  # Traverse through a chain of parents from a goal value until there is no parent.
  current = goal
  while current:
    yield current
    current = parents[current]
