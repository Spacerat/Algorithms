
from __future__ import print_function
from collections import namedtuple
from heap import BinHeap
from graph import Grid, Point, pathify

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
  return list(reversed(list(pathify(parents, goal))))

def main():
  # Try out the algorithm on a simple maze.
  themap = [
    [ 1,  1,  1,  1,  1, -1,  1,  1],
    [ 1,  1,  1,  1, 10, -1,  1,  1],
    [ 1,  1,  1,  1,  1,  1, -1,  1],
    [ 1,  1, -1, -1, -1, -1,  2,  1],
    [ 1,  1,  1,  1,  1,  1,  1,  1]]
  g = Grid(themap)


  path = Astar(g, Point(0, 0), Point(0, 6))
  g.printWithPath(path)

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
