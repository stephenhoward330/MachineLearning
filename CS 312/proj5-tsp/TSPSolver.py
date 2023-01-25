#!/usr/bin/python3

from PyQt5.QtCore import QLineF, QPointF




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		found_tour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not found_tour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(ncities)
			route = []
			# Now build the route using the random permutation
			for i in range(ncities):
				route.append(cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				found_tour = True
		end_time = time.time()
		results['cost'] = bssf.cost if found_tour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	def greedy(self, time_allowance=60.0):
		""" the greedy algorithm is naturally O(n^2), as it goes through each node, checking all others
			to find the lowest. We repeat it once starting at each node, so we get O(n^3) for time complexity.

			We will store only one TSPSolution object. This object is replaced as better solutions are found.
			A TSPSolution has a route (list) and a cost (number). So, our space complexity is O(n)."""
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0
		bssf = None
		start_time = time.time()
		for c in cities:  # start at every city O(n)
			current = c
			route = []
			found_tour = True
			for i in range(ncities):  # a route will contain all cities O(n)
				if (time.time()-start_time) > time_allowance:
					found_tour = False
					break
				route.append(current)
				if i == ncities-1:  # last one
					if route[-1].costTo(route[0]) is np.inf:
						found_tour = False
					break

				min_e = np.inf
				min_x = None
				for x in [y for y in cities if y not in route]:  # cities-route, simplifies to O(n)
					if current.costTo(x) < min_e:  # find closest city
						min_e = current.costTo(x)
						min_x = x

				if min_x is None:  # nowhere to go
					found_tour = False
					break
				current = min_x
			if found_tour is False:  # restart loop
				continue
			if bssf is None:  # first bssf
				count += 1
				bssf = TSPSolution(route)
			else:  # potentially replace bssf
				new_bssf = TSPSolution(route)
				if new_bssf.cost < bssf.cost:
					count += 1
					bssf = new_bssf

		end_time = time.time()
		results['cost'] = bssf.cost if bssf is not None else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
	def branchAndBound(self, time_allowance=60.0):
		""" For time complexity, we loop through our queue. This queue may end up with as many as 2^n
			objects in it. In each iteration, we call expand, which takes n^3 time. So, our total time complexity
			is O(2^n*n^3).

			Space complexity is the same at O(2^n*n^3). In the while loop we make up to n matrices of size n^2."""
		greedy_results = self.greedy()
		bssf = greedy_results['soln']

		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		count = 0  # num of BSSF changes
		max_q = 0  # max queue size
		total = 1  # total num of states created
		pruned = 0  # num of states pruned
		start_time = time.time()

		# make initial problem p
		m = []
		for a in cities:  # O(n)
			row = []
			for b in cities:  # O(n)
				row.append(a.costTo(b))
			m.append(row)
		# Creating a TSPState has O(n^2) space complexity, and reduction takes O(n^2) time
		p = TSPState(0, m, True, [cities[0]])

		s = [p]
		while len(s) > 0 and (time.time()-start_time) < time_allowance:  # max time complexity of 2^n
			if len(s) > max_q:
				max_q = len(s)
			prob = heapq.heappop(s)  # log(n) time
			if prob.lb >= bssf.cost:
				pruned += 1
				continue
			# expand prob into probs
			expanded_probs = self.expand(prob)  # O(n^3) space and time complexity
			total += len(expanded_probs)
			for p in expanded_probs:  # maximum of n expanded states
				if len(p.route) == ncities:
					new_bssf = TSPSolution(p.route)  # O(n) space complexity
					if new_bssf.cost < bssf.cost:
						count += 1
						bssf = new_bssf
					else:
						pruned += 1
				elif p.is_solvable and p.lb < bssf.cost:
					heapq.heappush(s, p)  # log(n) time
				else:
					pruned += 1
		pruned += len(s)

		end_time = time.time()
		results['cost'] = bssf.cost if bssf is not None else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = max_q
		results['total'] = total
		results['pruned'] = pruned
		return results

	def expand(self, prob):
		""" Total time complexity is O(n^3), since we might make n states that all take O(n^2) time to reduce.
			Space complexity is also O(n^3), since we might make n states all containing an n^2 matrix """
		cities = self._scenario.getCities()
		new_probs = []
		source_city = prob.route[-1]
		i = cities.index(source_city)
		for j in range(len(cities)):  # O(n)
			if prob.matrix[i][j] != np.inf:
				new_matrix = [row[:] for row in prob.matrix]  # O(n) time, O(n^2) space
				for k in range(len(cities)):  # set the source row to inf, O(n)
					new_matrix[i][k] = np.inf
				for k in range(len(cities)):  # set the destination column to inf, O(n)
					new_matrix[k][j] = np.inf
				new_matrix[j][i] = np.inf  # set the back path to inf
				is_solvable = False
				for k in range(len(cities)):  # check if it's solvable, O(n)
					if new_matrix[j][k] != np.inf:
						is_solvable = True
						break
				new_route = prob.route.copy()
				new_route.append(cities[j])
				# Creating a TSPState has O(n^2) space complexity, and reduction takes O(n^2) time
				new_prob = TSPState(prob.lb+prob.matrix[i][j], new_matrix, is_solvable, new_route)
				new_probs.append(new_prob)

		return new_probs  # n objects of size n^2, space complexity is O(n^3)


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
	def fancy(self, time_allowance=60.0):
		pass

		



