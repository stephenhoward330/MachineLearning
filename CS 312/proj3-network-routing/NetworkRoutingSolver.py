#!/usr/bin/python3


from CS312Graph import *
import time


class NetworkRoutingSolver:
    def __init__(self):
        pass

    def initializeNetwork(self, network):
        assert (type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        self.dest = destIndex

        # TODO: RETURN THE SHORTEST PATH FOR destIndex
        #       INSTEAD OF THE DUMMY SET OF EDGES BELOW
        #       IT'S JUST AN EXAMPLE OF THE FORMAT YOU'LL 
        #       NEED TO USE

        path_edges = []
        total_length = 0.0
        node = self.network.nodes[self.dest]
        while node.prev is not None:
            prev_node = node.prev

            # find edge between prev_node and node, must be one of three possibilities
            edge = prev_node.neighbors[0]
            if edge.dest.node_id != node.node_id:
                edge = prev_node.neighbors[1]
            if edge.dest.node_id != node.node_id:
                edge = prev_node.neighbors[2]

            path_edges.append((edge.src.loc, edge.dest.loc, '{:.0f}'.format(edge.length)))
            total_length += edge.length
            node = prev_node

        if total_length == 0.0:  # in this case, there is no path to our destination node
            total_length = float('inf')
        return {'cost': total_length, 'path': path_edges}

    def computeShortestPaths(self, srcIndex, use_heap):  # use_heap=False
        self.source = srcIndex
        t1 = time.time()

        # TODO: RUN DIJKSTRA'S TO DETERMINE SHORTEST PATHS.
        #       ALSO, STORE THE RESULTS FOR THE SUBSEQUENT
        #       CALL TO getShortestPath(dest_index)

        if use_heap:  # min heap implementation -- O(n log n) time, O(n) space
            for u in self.network.nodes:  # O(n)
                u.dist = float('inf')
                u.prev = None
            self.network.nodes[self.source].dist = 0.0
            q = self.h_make_queue(self.network.nodes)  # O(n) time and space
            while len(q) > 0:  # we will call this once for every node, so O(n), or a total of O(n log n) time
                u = self.h_delete_min(q)  # O(log n)
                for e in u.neighbors:  # only 3 neighbors, so reduces to O(1), for a total of O(log n) time
                    w = e.length
                    v = e.dest
                    if u.dist + w < v.dist:
                        v.dist = u.dist + w
                        v.prev = u
                        self.h_decrease_key(q, v)  # O(log n)
        else:  # unsorted array implementation -- O(n^2) time, O(n) space
            for u in self.network.nodes:  # O(n)
                u.dist = float('inf')
                u.prev = None
            self.network.nodes[self.source].dist = 0.0
            q = self.u_make_queue(self.network.nodes)  # O(n) time and space
            while len(q) > 0:  # we will call this once for every node, so O(n), or a total of O(n^2) time
                u = self.u_delete_min(q)  # O(n)
                for e in u.neighbors:  # only 3 neighbors, so reduces to O(1)
                    w = e.length
                    v = e.dest
                    if u.dist + w < v.dist:
                        v.dist = u.dist + w
                        v.prev = u
                        # self.u_decrease_key(q, v) does nothing for an unsorted array; is not needed

        t2 = time.time()
        return t2 - t1

    def h_make_queue(self, nodes):  # total time complexity: O(n), space complexity O(n)
        new_nodes = nodes.copy()  # O(n) time and space

        # make a hash table for mapping nodes to their min-heap indices
        self.table = dict(key="value")
        for i in range(len(new_nodes)):  # O(n) time and space
            self.table[new_nodes[i]] = i
        self.table.pop("key")

        self.h_decrease_key(new_nodes, new_nodes[self.source])  # O(log n) time
        return new_nodes

    def h_delete_min(self, q):  # time complexity O(log n), space complexity O(1)
        x = q[0]  # first element in heap
        y = q[len(q) - 1]  # last element in heap
        q.pop()  # remove last element -- O(1)

        # put last element on top and trickle down
        if len(q) > 0:
            self.sift_down(q, y, 0)  # O(log n)
        self.table.pop(x)  # average O(1)
        return x

    def sift_down(self, q, y, i):  # time complexity O(log n), space complexity O(1)
        # helper for h_delete_min()

        # trickle down
        c = self.min_child(q, i)
        while c is not None and q[c].dist < y.dist:  # this loop will be called at most log(n) times
            q[i] = q[c]
            self.table[q[i]] = i
            i = c
            c = self.min_child(q, i)
        q[i] = y
        self.table[y] = i

    def min_child(self, q, i):  # time complexity O(1)
        # helper for h_delete_min(); determines which of the (up to) two children is smallest
        if (2 * (i + 1)) - 1 >= len(q):  # no children
            return None
        elif (2 * (i + 1)) == len(q):  # only left child
            return (2 * (i + 1)) - 1
        else:  # two children
            if q[(2 * (i + 1)) - 1].dist <= q[2 * (i + 1)].dist:  # left child is lower or equal than right child
                return (2 * (i + 1)) - 1
            else:  # right child is lower than left child
                return 2 * (i + 1)

    def h_decrease_key(self, q, v):  # time complexity O(log n), space complexity O(1)
        # find node to bubble up
        child_index = self.table[v]  # O(1)
        if child_index is None:  # do nothing if node is not in the table
            return
        parent_index = (child_index - 1) // 2
        parent = q[parent_index]

        # bubble up
        while child_index != 0 and parent.dist > v.dist:  # this loop will be called at most log(n) times
            q[child_index] = parent
            self.table[parent] = child_index
            child_index = parent_index
            parent_index = (child_index - 1) // 2
            parent = q[parent_index]
        q[child_index] = v
        self.table[v] = child_index

    def u_make_queue(self, nodes):  # time complexity O(n), space complexity O(n)
        return nodes.copy()  # O(n)

    def u_delete_min(self, q):  # time complexity O(n), space complexity O(1)
        min_length = float('inf')
        winning_index = 0

        # find the node with the lowest distance
        for i in range(len(q)):  # O(n)
            if q[i].dist < min_length:
                min_length = q[i].dist
                winning_index = i
        winner_id = q[winning_index].node_id

        # remove the node with the lowest distance
        q.pop(winning_index)  # worst case O(n)
        return self.network.nodes[winner_id]  # return it
