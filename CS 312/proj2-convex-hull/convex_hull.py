# this is 4-5 seconds slower on 1000000 points than Ryan's desktop...  Why?


from PyQt5.QtCore import QLineF, QPointF, QThread, pyqtSignal



import time



class ConvexHullSolverThread(QThread):
    def __init__( self, unsorted_points,demo):
        self.points = unsorted_points                    
        self.pause = demo
        QThread.__init__(self)

    def __del__(self):
        self.wait()

    # These two signals are used for interacting with the GUI.
    show_hull    = pyqtSignal(list,tuple)
    display_text = pyqtSignal(str)

    # Some additional thread signals you can implement and use for debugging,
    # if you like
    show_tangent = pyqtSignal(list,tuple)
    erase_hull = pyqtSignal(list)
    erase_tangent = pyqtSignal(list)
                    

    def set_points( self, unsorted_points, demo):
        self.points = unsorted_points
        self.demo   = demo


    def run(self):
        assert( type(self.points) == list and type(self.points[0]) == QPointF )

        n = len(self.points)
        print( 'Computing Hull for set of {} points'.format(n) )

        t1 = time.time()
        # SORT THE POINTS BY INCREASING X-VALUE
        # We say that sorting has time complexity O(n log n)
        # sorted has space complexity of O(n)
        sorted_points = sorted(self.points, key=self.comparison_fn)
        t2 = time.time()
        print('Time Elapsed (Sorting): {:3.3f} sec'.format(t2-t1))

        t3 = time.time()
        # COMPUTE THE CONVEX HULL USING DIVIDE AND CONQUER
        # Our total recursive Hull algorithm has time complexity O(n log n) from the master theorem with a,b = 2, d = 1
        # Space complexity is O(n log n) since the most that will be stored on one level is O(n), and
        #   we have at most O(log n) levels
        top_hull, bottom_hull = self.dac_hull(sorted_points)
        top_hull.extend(bottom_hull[1:len(bottom_hull)-1])  # put the full hull in top_hull
        t4 = time.time()

        USE_DUMMY = False
        if USE_DUMMY:
            # This is a dummy polygon of the first 3 unsorted points
            polygon = [QLineF(self.points[i],self.points[(i+1)%3]) for i in range(3)]
            
            # When passing lines to the display, pass a list of QLineF objects.
            # Each QLineF object can be created with two QPointF objects
            # corresponding to the endpoints
            assert( type(polygon) == list and type(polygon[0]) == QLineF )

            # Send a signal to the GUI thread with the hull and its color
            self.show_hull.emit(polygon,(0,255,0))

        else:
            # TODO: PASS THE CONVEX HULL LINES BACK TO THE GUI FOR DISPLAY
            polygon = [QLineF(top_hull[i], top_hull[(i + 1) % len(top_hull)]) for i in range(len(top_hull))]
            assert (type(polygon) == list and type(polygon[0]) == QLineF)

            self.show_hull.emit(polygon, (0, 255, 0))

        # Send a signal to the GUI thread with the time used to compute the hull
        self.display_text.emit("Time Elapsed (Convex Hull): {:3.3f} sec".format(t4-t3))
        print('Time Elapsed (Convex Hull): {:3.3f} sec'.format(t4-t3))

    def comparison_fn(self, elem):
        return elem.x()

    def dac_hull(self, point_array):
        # base cases
        if len(point_array) == 1:
            return point_array, point_array
        if len(point_array) == 2:
            upper_hull = [point_array[0], point_array[1]]
            lower_hull = [point_array[1], point_array[0]]
            return upper_hull, lower_hull

        # recursive calls
        left_upper_hull, left_lower_hull = self.dac_hull(point_array[:len(point_array)//2])
        right_upper_hull, right_lower_hull = self.dac_hull(point_array[len(point_array)//2:])

        # prepare for recombine
        # the whole recombine takes a total of O(n) time
        left_upper_index = len(left_upper_hull)-1
        left_lower_index = 0
        right_upper_index = 0
        right_lower_index = len(right_lower_hull)-1

        # draw some lines
        if self.pause:
            test_left_hull = left_upper_hull.copy()
            test_left_hull.extend(left_lower_hull[1:len(left_lower_hull) - 1])
            polygon = [QLineF(test_left_hull[i], test_left_hull[(i + 1) % len(test_left_hull)]) for i in range(len(test_left_hull))]
            assert (type(polygon) == list and type(polygon[0]) == QLineF)
            self.show_hull.emit(polygon, (0, 0, 255))

            test_right_hull = right_upper_hull.copy()
            test_right_hull.extend(right_lower_hull[1:len(right_lower_hull) - 1])
            polygon = [QLineF(test_right_hull[i], test_right_hull[(i + 1) % len(test_right_hull)]) for i in
                       range(len(test_right_hull))]
            assert (type(polygon) == list and type(polygon[0]) == QLineF)
            self.show_hull.emit(polygon, (0, 0, 255))

        # RECOMBINE TOP O(n) time complexity
        big_changes = True
        slope = self.get_slope(right_upper_hull[right_upper_index], left_upper_hull[left_upper_index])

        # show tangent lines
        if self.pause:
            self.show_tangent.emit([QLineF(right_upper_hull[right_upper_index], left_upper_hull[left_upper_index])],
                                   (255, 0, 0))

        while big_changes:
            changes1, changes2, big_changes = True, True, False
            while changes1:
                changes1 = False
                new_slope = self.get_slope(right_upper_hull[(right_upper_index+1) % len(right_upper_hull)], left_upper_hull[left_upper_index])
                if new_slope > slope:
                    slope = new_slope
                    right_upper_index += 1

                    # show possible tangent line
                    if self.pause:
                        self.show_tangent.emit(
                            [QLineF(right_upper_hull[right_upper_index], left_upper_hull[left_upper_index])],
                            (255, 0, 0))

                    changes1, big_changes = True, True

            slope = self.get_slope(right_upper_hull[right_upper_index], left_upper_hull[left_upper_index])

            while changes2:
                changes2 = False
                new_slope = self.get_slope(right_upper_hull[right_upper_index], left_upper_hull[(left_upper_index-1) % len(left_upper_hull)])
                if new_slope < slope:
                    slope = new_slope
                    left_upper_index -= 1

                    # show possible tangent line
                    if self.pause:
                        self.show_tangent.emit(
                            [QLineF(right_upper_hull[right_upper_index], left_upper_hull[left_upper_index])],
                            (255, 0, 0))

                    changes2, big_changes = True, True

        # RECOMBINE BOTTOM O(n) time complexity
        big_changes = True
        slope = self.get_slope(right_lower_hull[right_lower_index], left_lower_hull[left_lower_index])

        # show tangent lines
        if self.pause:
            self.show_tangent.emit([QLineF(right_lower_hull[right_lower_index], left_lower_hull[left_lower_index])],
                                   (255, 0, 0))

        while big_changes:
            changes1, changes2, big_changes = True, True, False
            while changes1:
                changes1 = False
                new_slope = self.get_slope(right_lower_hull[(right_lower_index - 1) % len(right_lower_hull)], left_lower_hull[left_lower_index])
                if new_slope < slope:
                    slope = new_slope
                    right_lower_index -= 1

                    # show possible tangent line
                    if self.pause:
                        self.show_tangent.emit(
                            [QLineF(right_lower_hull[right_lower_index], left_lower_hull[left_lower_index])], (255, 0, 0))

                    changes1, big_changes = True, True

            slope = self.get_slope(right_lower_hull[right_lower_index], left_lower_hull[left_lower_index])

            while changes2:
                changes2 = False
                new_slope = self.get_slope(right_lower_hull[right_lower_index], left_lower_hull[(left_lower_index + 1) % len(left_lower_hull)])
                if new_slope > slope:
                    slope = new_slope
                    left_lower_index += 1

                    # show possible tangent line
                    if self.pause:
                        self.show_tangent.emit(
                            [QLineF(right_lower_hull[right_lower_index], left_lower_hull[left_lower_index])], (255, 0, 0))

                    changes2, big_changes = True, True

        # combine the hulls and return
        left_upper_hull = left_upper_hull[:left_upper_index+1]
        right_upper_hull = right_upper_hull[right_upper_index:]
        left_lower_hull = left_lower_hull[left_lower_index:]
        right_lower_hull = right_lower_hull[:right_lower_index+1]

        # add upper hulls to lower hulls
        # extend has O(n) time complexity
        left_upper_hull.extend(right_upper_hull)
        right_lower_hull.extend(left_lower_hull)
        # remove intermediate lines
        self.erase_hull.emit([])

        return left_upper_hull, right_lower_hull

    # this has O(1) time and space complexity
    def get_slope(self, point1, point2):
        return (point2.y()-point1.y())/(point2.x()-point1.x())
