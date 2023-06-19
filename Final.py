# class Solution:
#     def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
#         jobs = sorted(zip(startTime, endTime, profit))
#         n = len(jobs)
#         startTime.sort()
#
#         dp = [0 for _ in range(n + 1)]
#         for i in range(n - 1, -1, -1):
#             j = bisect.bisect_left(startTime, jobs[i][1])
#             dp[i] = max(dp[i + 1], dp[j] + jobs[i][2])
#         return dp[0]

# class Solution:
#     def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
#         jobs = sorted(zip(startTime, endTime, profit))
#         n = len(jobs)
#
#         def helper(val: int) -> int:
#             l, r = 0, n - 1
#             while l <= r:
#                 m = (l + r) >> 1
#                 s, e, p = jobs[m]
#                 if s < val: l = m + 1
#                 else: r = m - 1
#             return l
#
#         dp = [0 for _ in range(n + 1)]
#         for i in range(n - 1, -1, -1):
#             j = helper(jobs[i][1])
#             dp[i] = max(dp[i + 1], dp[j] + jobs[i][2])
#         return dp[0]
# Python3 program for weighted job scheduling
# using Dynamic Programming

# Importing the following module to sort array
# based on our custom comparison function
from functools import cmp_to_key

# A job has start time, finish time and profit


class Job:

	def __init__(self, start, finish, profit):

		self.start = start
		self.finish = finish
		self.profit = profit

# A utility function that is used for sorting
# events according to finish time

def jobComparator(s1, s2):

	return s1.finish < s2.finish

# Find the latest job (in sorted array) that
# doesn't conflict with the job[i]. If there
# is no compatible job, then it returns -1


def latestNonConflict(arr, i):

	for j in range(i - 1, -1, -1):
		if arr[j].finish <= arr[i - 1].start:
			return j

	return -1

# The main function that returns the maximum possible
# profit from given array of jobs


def findMaxProfit(arr, n):

	# Sort jobs according to finish time
	arr = sorted(arr, key=cmp_to_key(jobComparator))

	# Create an array to store solutions of subproblems.
	# table[i] stores the profit for jobs till arr[i]
	# (including arr[i])
	table = [None] * n
	table[0] = arr[0].profit

	# Fill entries in M[] using recursive property
	for i in range(1, n):

		# Find profit including the current job
		inclProf = arr[i].profit
		l = latestNonConflict(arr, i)

		if l != -1:
			inclProf += table[l]

		# Store maximum of including and excluding
		table[i] = max(inclProf, table[i - 1])

	# Store result and free dynamic memory
	# allocated for table[]
	result = table[n - 1]

	return result


# Driver code
values = [(3, 10, 20), (1, 2, 50),
		(6, 19, 100), (2, 100, 200)]
arr = []
for i in values:
	arr.append(Job(i[0], i[1], i[2]))

n = len(arr)

print("The optimal profit is", findMaxProfit(arr, n))

# This code is contributed by Kevin Joshi

from sys import maxint

def maxSubArraySum(a, size):
    max_so_far = -maxint - 1
    max_ending_here = 0

    for i in range(0, size):
        max_ending_here = max_ending_here + a[i]
        if (max_so_far < max_ending_here):
            max_so_far = max_ending_here

        if max_ending_here < 0:
            max_ending_here = 0
    return max_so_far

import sys

# Define a function to find the maximum subarray sum


def maxSubArraySum(arr):
	# Base case: when there is only one element in the array
	if len(arr) == 1:
		return arr[0]

	# Recursive case: divide the problem into smaller sub-problems
	m = len(arr) // 2

	# Find the maximum subarray sum in the left half
	left_max = maxSubArraySum(arr[:m])

	# Find the maximum subarray sum in the right half
	right_max = maxSubArraySum(arr[m:])

	# Find the maximum subarray sum that crosses the middle element
	left_sum = -sys.maxsize - 1
	right_sum = -sys.maxsize - 1
	sum = 0

	# Traverse the array from the middle to the right
	for i in range(m, len(arr)):
		sum += arr[i]
		right_sum = max(right_sum, sum)

	sum = 0

	# Traverse the array from the middle to the left
	for i in range(m - 1, -1, -1):
		sum += arr[i]
		left_sum = max(left_sum, sum)

	cross_max = left_sum + right_sum

	# Return the maximum of the three subarray sums
	return max(cross_max, max(left_max, right_max))


# Example usage
arr = [-2, -3, 4, -1, -2, 1, 5, -3]
max_sum = maxSubArraySum(arr)
print("Maximum contiguous sum is", max_sum)


# This is the memoization approach of
# 0 / 1 Knapsack in Python in simple
# we can say recursion + memoization = DP


def knapsack(wt, val, W, n):
	# base conditions
	if n == 0 or W == 0:
		return 0
	if t[n][W] != -1:
		return t[n][W]

	# choice diagram code
	if wt[n - 1] <= W:
		t[n][W] = max(
			val[n - 1] + knapsack(
				wt, val, W - wt[n - 1], n - 1),
			knapsack(wt, val, W, n - 1))
		return t[n][W]
	elif wt[n - 1] > W:
		t[n][W] = knapsack(wt, val, W, n - 1)
		return t[n][W]


# Driver code
if __name__ == '__main__':
	profit = [60, 100, 120]
	weight = [10, 20, 30]
	W = 50
	n = len(profit)

	# We initialize the matrix with -1 at first.
	t = [[-1 for i in range(W + 1)] for j in range(n + 1)]
	print(knapsack(weight, profit, W, n))

# This code is contributed by Prosun Kumar Sarkar

# Python program to find the
# regression line

# Function to calculate b
def calculateB(x, y, n):
	# sum of array x
	sx = sum(x)

	# sum of array y
	sy = sum(y)

	# for sum of product of x and y
	sxsy = 0

	# sum of square of x
	sx2 = 0

	for i in range(n):
		sxsy += x[i] * y[i]
		sx2 += x[i] * x[i]
	b = (n * sxsy - sx * sy) / (n * sx2 - sx * sx)
	return b


# Function to find the
# least regression line
def leastRegLine(X, Y, n):
	# Finding b
	b = calculateB(X, Y, n)
	meanX = int(sum(X) / n)
	meanY = int(sum(Y) / n)

	# Calculating a
	a = meanY - b * meanX

	# Printing regression line
	print("Regression line:")
	print("Y = ", '%.3f' % a, " + ", '%.3f' % b, "*X", sep="")


# Driver code

# Statistical data
X = [95, 85, 80, 70, 60]
Y = [90, 80, 70, 65, 60]
n = len(X)
leastRegLine(X, Y, n)

# This code is contributed by avanitrachhadiya2155
# !/bin/python3
"""
Origin: https://www.geeksforgeeks.org/sequence-alignment-problem/
Converted from C++ solution to Python3

Algorithm type / application: Bioinformatics


Python Requirements:
	numpy

"""
import numpy as np


def get_minimum_penalty(x: str, y: str, pxy: int, pgap: int):
	"""
	Function to find out the minimum penalty

	:param x: pattern X
	:param y: pattern Y
	:param pxy: penalty of mis-matching the characters of X and Y
	:param pgap: penalty of a gap between pattern elements
	"""

	# initializing variables
	i = 0
	j = 0

	# pattern lengths
	m = len(x)
	n = len(y)

	# table for storing optimal substructure answers
	dp = np.zeros([m + 1, n + 1], dtype=int)  # int dp[m+1][n+1] = {0};

	# initialising the table
	dp[0:(m + 1), 0] = [i * pgap for i in range(m + 1)]
	dp[0, 0:(n + 1)] = [i * pgap for i in range(n + 1)]

	# calculating the minimum penalty
	i = 1
	while i <= m:
		j = 1
		while j <= n:
			if x[i - 1] == y[j - 1]:
				dp[i][j] = dp[i - 1][j - 1]
			else:
				dp[i][j] = min(dp[i - 1][j - 1] + pxy,
							   dp[i - 1][j] + pgap,
							   dp[i][j - 1] + pgap)
			j += 1
		i += 1

	# Reconstructing the solution
	l = n + m  # maximum possible length
	i = m
	j = n

	xpos = l
	ypos = l

	# Final answers for the respective strings
	xans = np.zeros(l + 1, dtype=int)
	yans = np.zeros(l + 1, dtype=int)

	while not (i == 0 or j == 0):
		# print(f"i: {i}, j: {j}")
		if x[i - 1] == y[j - 1]:
			xans[xpos] = ord(x[i - 1])
			yans[ypos] = ord(y[j - 1])
			xpos -= 1
			ypos -= 1
			i -= 1
			j -= 1
		elif (dp[i - 1][j - 1] + pxy) == dp[i][j]:

			xans[xpos] = ord(x[i - 1])
			yans[ypos] = ord(y[j - 1])
			xpos -= 1
			ypos -= 1
			i -= 1
			j -= 1

		elif (dp[i - 1][j] + pgap) == dp[i][j]:
			xans[xpos] = ord(x[i - 1])
			yans[ypos] = ord('_')
			xpos -= 1
			ypos -= 1
			i -= 1

		elif (dp[i][j - 1] + pgap) == dp[i][j]:
			xans[xpos] = ord('_')
			yans[ypos] = ord(y[j - 1])
			xpos -= 1
			ypos -= 1
			j -= 1

	while xpos > 0:
		if i > 0:
			i -= 1
			xans[xpos] = ord(x[i])
			xpos -= 1
		else:
			xans[xpos] = ord('_')
			xpos -= 1

	while ypos > 0:
		if j > 0:
			j -= 1
			yans[ypos] = ord(y[j])
			ypos -= 1
		else:
			yans[ypos] = ord('_')
			ypos -= 1

	# Since we have assumed the answer to be n+m long,
	# we need to remove the extra gaps in the starting
	# id represents the index from which the arrays
	# xans, yans are useful
	id = 1
	i = l
	while i >= 1:
		if (chr(yans[i]) == '_') and chr(xans[i]) == '_':
			id = i + 1
			break

		i -= 1

	# Printing the final answer
	print(f"Minimum Penalty in aligning the genes = {dp[m][n]}")
	print("The aligned genes are:")
	# X
	i = id
	x_seq = ""
	while i <= l:
		x_seq += chr(xans[i])
		i += 1
	print(f"X seq: {x_seq}")

	# Y
	i = id
	y_seq = ""
	while i <= l:
		y_seq += chr(yans[i])
		i += 1
	print(f"Y seq: {y_seq}")


def test_get_minimum_penalty():
	"""
	Test the get_minimum_penalty function
	"""
	# input strings
	gene1 = "AGGGCT"
	gene2 = "AGGCA"

	# initialising penalties of different types
	mismatch_penalty = 3
	gap_penalty = 2

	# calling the function to calculate the result
	get_minimum_penalty(gene1, gene2, mismatch_penalty, gap_penalty)


test_get_minimum_penalty()

# This code is contributed by wilderchirstopher.
# A Top-Down DP implementation of LCS problem

# Returns length of LCS for X[0..m-1], Y[0..n-1]
def lcs(X, Y, m, n, dp):

	if (m == 0 or n == 0):
		return 0

	if (dp[m][n] != -1):
		return dp[m][n]

	if X[m - 1] == Y[n - 1]:
		dp[m][n] = 1 + lcs(X, Y, m - 1, n - 1, dp)
		return dp[m][n]

	dp[m][n] = max(lcs(X, Y, m, n - 1, dp),lcs(X, Y, m - 1, n, dp))
	return dp[m][n]

# Driver code

X = "AGGTAB"
Y = "GXTXAYB"

m = len(X)
n = len(Y)
dp = [[-1 for i in range(n + 1)]for j in range(m + 1)]

print(f"Length of LCS is {lcs(X, Y, m, n, dp)}")

# This code is contributed by shinjanpatra

# Python3 program for Bellman-Ford's
# single source shortest path algorithm.
from sys import maxsize

# The main function that finds shortest
# distances from src to all other vertices
# using Bellman-Ford algorithm. The function
# also detects negative weight cycle
# The row graph[i] represents i-th edge with
# three values u, v and w.
def BellmanFord(graph, V, E, src):

	# Initialize distance of all vertices as infinite.
	dis = [maxsize] * V

	# initialize distance of source as 0
	dis[src] = 0

	# Relax all edges |V| - 1 times. A simple
	# shortest path from src to any other
	# vertex can have at-most |V| - 1 edges
	for i in range(V - 1):
		for j in range(E):
			if dis[graph[j][0]] + \
				graph[j][2] < dis[graph[j][1]]:
				dis[graph[j][1]] = dis[graph[j][0]] + \
									graph[j][2]

	# check for negative-weight cycles.
	# The above step guarantees shortest
	# distances if graph doesn't contain
	# negative weight cycle. If we get a
	# shorter path, then there is a cycle.
	for i in range(E):
		x = graph[i][0]
		y = graph[i][1]
		weight = graph[i][2]
		if dis[x] != maxsize and dis[x] + \
						weight < dis[y]:
			print("Graph contains negative weight cycle")

	print("Vertex Distance from Source")
	for i in range(V):
		print("%d\t\t%d" % (i, dis[i]))

# Driver Code
if __name__ == "__main__":
	V = 5 # Number of vertices in graph
	E = 8 # Number of edges in graph

	# Every edge has three values (u, v, w) where
	# the edge is from vertex u to v. And weight
	# of the edge is w.
	graph = [[0, 1, -1], [0, 2, 4], [1, 2, 3],
			[1, 3, 2], [1, 4, 2], [3, 2, 5],
			[3, 1, 1], [4, 3, -3]]
	BellmanFord(graph, V, E, 0)

# This code is contributed by
# sanjeev2552
# A Python3 program to check if a graph contains negative
# weight cycle using Bellman-Ford algorithm. This program
# works only if all vertices are reachable from a source
# vertex 0.

# a structure to represent a weighted edge in graph
class Edge:

	def __init__(self):
		self.src = 0
		self.dest = 0
		self.weight = 0


# a structure to represent a connected, directed and
# weighted graph
class Graph:

	def __init__(self):
		# V. Number of vertices, E. Number of edges
		self.V = 0
		self.E = 0

		# graph is represented as an array of edges.
		self.edge = None


# Creates a graph with V vertices and E edges
def createGraph(V, E):
	graph = Graph()
	graph.V = V;
	graph.E = E;
	graph.edge = [Edge() for i in range(graph.E)]
	return graph;


# The main function that finds shortest distances
# from src to all other vertices using Bellman-
# Ford algorithm. The function also detects
# negative weight cycle
def isNegCycleBellmanFord(graph, src):
	V = graph.V;
	E = graph.E;
	dist = [1000000 for i in range(V)];
	dist[src] = 0;

	# Step 2: Relax all edges |V| - 1 times.
	# A simple shortest path from src to any
	# other vertex can have at-most |V| - 1
	# edges
	for i in range(1, V):
		for j in range(E):

			u = graph.edge[j].src;
			v = graph.edge[j].dest;
			weight = graph.edge[j].weight;
			if (dist[u] != 1000000 and dist[u] + weight < dist[v]):
				dist[v] = dist[u] + weight;

	# Step 3: check for negative-weight cycles.
	# The above step guarantees shortest distances
	# if graph doesn't contain negative weight cycle.
	# If we get a shorter path, then there
	# is a cycle.
	for i in range(E):

		u = graph.edge[i].src;
		v = graph.edge[i].dest;
		weight = graph.edge[i].weight;
		if (dist[u] != 1000000 and dist[u] + weight < dist[v]):
			return True;

	return False;


# Driver program to test above functions
if __name__ == '__main__':

	# Let us create the graph given in above example
	V = 5;  # Number of vertices in graph
	E = 8;  # Number of edges in graph
	graph = createGraph(V, E);

	# add edge 0-1 (or A-B in above figure)
	graph.edge[0].src = 0;
	graph.edge[0].dest = 1;
	graph.edge[0].weight = -1;

	# add edge 0-2 (or A-C in above figure)
	graph.edge[1].src = 0;
	graph.edge[1].dest = 2;
	graph.edge[1].weight = 4;

	# add edge 1-2 (or B-C in above figure)
	graph.edge[2].src = 1;
	graph.edge[2].dest = 2;
	graph.edge[2].weight = 3;

	# add edge 1-3 (or B-D in above figure)
	graph.edge[3].src = 1;
	graph.edge[3].dest = 3;
	graph.edge[3].weight = 2;

	# add edge 1-4 (or A-E in above figure)
	graph.edge[4].src = 1;
	graph.edge[4].dest = 4;
	graph.edge[4].weight = 2;

	# add edge 3-2 (or D-C in above figure)
	graph.edge[5].src = 3;
	graph.edge[5].dest = 2;
	graph.edge[5].weight = 5;

	# add edge 3-1 (or D-B in above figure)
	graph.edge[6].src = 3;
	graph.edge[6].dest = 1;
	graph.edge[6].weight = 1;

	# add edge 4-3 (or E-D in above figure)
	graph.edge[7].src = 4;
	graph.edge[7].dest = 3;
	graph.edge[7].weight = -3;

	if (isNegCycleBellmanFord(graph, 0)):
		print("Yes")
	else:
		print("No")

# This code is contributed by pratham76


