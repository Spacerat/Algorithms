import math
from itertools import islice

def merge(lista, listb):
	""" Merge two sorted lists into one sorted list.
	This operation has time a complexity of O(N), since it processes each list item once."""
	out = []
	i_a = 0
	i_b = 0
	num_items = len(lista) + len(listb)
	while i_a + i_b < num_items:
		# Examine the front of each list and append the smaller value to the output.
		if i_a < len(lista) and (i_b >= len(listb) or lista[i_a] <= listb[i_b]):
			out.append(lista[i_a])
			i_a+=1
		else:
			out.append(listb[i_b])
			i_b+=1
	return out


def mergesort(l):
	""" Recursively split a list into smaller lists until each list has size=1
	Then 'merge' adjacent lists, thereby sorting them until the full list is sorted.

	Since mergesort divides the list in half each iteration, it multiplies the complexity
	of 'merge' by LogN, thus its complexity is O(N LogN)

	"""

	# A list with length 1 is already sorted.
	if len(l) == 1:
		return l

	# Split the list in half
	halfway = math.floor(len(l)/2)

	# Slicing the list is also O(N). It's not exactly ideal from a performance
	# perspective but it doesn't increase the big O time complexity.
	left = l[:halfway]
	right = l[halfway:]

	# Mergesort each half
	left = mergesort(left)
	right = mergesort(right)

	# Merge the sorted halves
	return merge(left, right)


def main():
	print(mergesort([6, 5, 3, 1, 8, 7, 2, 4, 12, -1, 15]))

if __name__ == '__main__':
	main()

