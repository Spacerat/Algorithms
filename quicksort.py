import math

def partition(l, start, end, mid_pivot=True):
    if start >= end: return
    # A pivot value is selected from the range, here we take the middle,
    # but other schemes are possible.
    pivot = l[math.floor(start + (end - start) / 2)] if mid_pivot else l[start]
    # Indices i and j begin at the start/end points and finish at a mid point
    i, j = start, end
    while i <= j:
        # They are moved towards a mid point until they come across values
        # which are the wrong way round relative to the 'pivot' value.
        while l[i] < pivot: i += 1
        while l[j] > pivot: j -= 1
        if i <= j:
            # The values are swapped
            l[i], l[j] = l[j], l[i]
            # and the indices move on
            i, j = i + 1, j - 1
    # Once i and j meet, values[i:] will be greater than the pivot
    # and values[:j] will be lower than the pivot. 
    return i, j

def quicksort_part(l, start, end, mid_pivot=True):
    if start >= end: return
    i, j = partition(l, start, end, mid_pivot=mid_pivot)
    quicksort_part(l, start, j)
    quicksort_part(l, i, end, mid_pivot=mid_pivot)

def quicksort(l, mid_pivot=True):
    # In-place quicksort. Best-case time complexity is O(NLog N) but worst case is O(N^2)
    # However, quicksort is often faster than mergesort in practice since it can operate
    # in-place (so, O(N) space and no copying or new allocation is needed).
    quicksort_part(l, 0, len(l)-1, mid_pivot=mid_pivot)
    return l

def main():
    print(quicksort([1, 9, 3, -2, 7, 5, 15, 2, 5, 13, -5]))

if __name__ == '__main__':
    main()

