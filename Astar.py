
from __future__ import print_function
from collections import namedtuple
from heap import BinHeap


class BaseGraph():
  def neighbors(self, node):
    return []
  def cost(self, node1, node2):
    return 1
  def heuristic(self, node1, node2):
    return 1

Point = namedtuple('Point', ['row','col'])

class Grid(BaseGraph):
  def __init__(self, gridmap):
    self.grid = gridmap

  def neighbors(self, point):
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
    # assume nodes are adjacent
    return self.grid[node2.row][node2.col]

  def heuristic(self, node1, node2):
    return abs(node2.row - node1.row) + abs(node2.col - node1.col)
        
def Astar(graph, start, goal, priority_with_cost=True, priority_with_heuristc=True):
  visited = {}
  parent = {}
  cost_to_start = {}

  openset = BinHeap(maxheap = True)
  openset.push(0, start)
  parent[start] = None
  cost_to_start[start] = 0
  num_visits = 0

  while not openset.empty():
    current = openset.popval()

    if current == goal:
      break

    for next in graph.neighbors(current):
      new_cost = cost_to_start[current] + graph.cost(current, next)
      if next not in cost_to_start or new_cost < cost_to_start[next]:
        cost_to_start[next] = new_cost
        priority = new_cost if priority_with_cost else 0
        if priority_with_heuristc:
          priority += graph.heuristic(goal, next)

        openset.push(priority, next)
        parent[next] = current
        num_visits +=1

  print(num_visits, "vists")

  return reversed(list(pathify(parent, goal)))


def pathify(parent, goal):
  current = goal
  while current:
    yield current
    current = parent[current]



def main():
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


# OPEN = priority queue containing START
# CLOSED = empty set
# while lowest rank in OPEN is not the GOAL:
#   current = remove lowest rank item from OPEN
#   add current to CLOSED
#   for neighbors of current:
#     cost = g(current) + movementcost(current, neighbor)
#     if neighbor in OPEN and cost less than g(neighbor):
#       remove neighbor from OPEN, because new path is better
#     if neighbor in CLOSED and cost less than g(neighbor): **
#       remove neighbor from CLOSED
#     if neighbor not in OPEN and neighbor not in CLOSED:
#       set g(neighbor) to cost
#       add neighbor to OPEN
#       set priority queue rank to g(neighbor) + h(neighbor)
#       set neighbor's parent to current

# reconstruct reverse path from goal to start
# by following parent pointers