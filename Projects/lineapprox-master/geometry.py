"""
Geometry:  Points and polylines, including
approximation using the Ramer-Douglas-Peucker
algorithm.   

This is a 'model' component and should not directly 
contain any graphics code. All coordinates are in the 
model coordinate system.  The unit of the coordinate 
system can be any metric (cm, meters, whatever), but 
it must be the same unit in the x and y dimension 
so that the the usual distance formula holds.  Note that 
this IS true of UTM coordinates but it is NOT true of 
latitude and longitude. 

Class PolyLine is the representation of a sequence of 
points.  Includes hooks for a view component. 

Events generated (for view components): 
   "trial_approx" with options = { "p1": (x,y), "p2": (x,y) }
   "final_approx_seg" with options = { "p1": (x,y), "p2": (x,y) }

A trial approximation (event "trial_approx") is a segment that may 
be further subdivided.  A final approximation (event "final_approx_seg")
is a segment that will be incorporated in the line approximation with 
no further refinement.  

Author: Henzi Kou

"""
import math

from typing import List, Tuple, Any
from numbers import Number

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Point(object):
    """Integer Cartesian coordinates. Immutable.
    x and y coordinates are public, read-only.
    """
    pass

class PolyLine(object):
    """A polyline is a sequence of points.
    All fields are private.
    """
    # Part 1:
    #   by calling corresponding methods in the _points field

    def __init__(self, points):
        self._listeners = [ ]
        self.tolerance = 0
        self.points = points
        self.approx = points

    def __repr__(self):
        return "PolyLine[Point({}, {})]".format(self.points, self.approx)

    def append(self, tolerance):
        self.tolerance = tolerance
        self.approx = []
        self.approximate(0, len(self.points) - 1)
        self.approx.append(self.points[-1])
        return

    # ----- List protocol emulation --------
    # We want the PolyLine object to act like other Python
    # sequence objects (lists, tuples, etc).  These methods
    # should act exactly like their counterpoints in class
    # 'list'.



    # ---- End of list protocol emulation ---------

    # ---- Connection to graphics in Model-View-Controller (MVC) style;
    #      We will talk about this in week 2.  Don't change this section
    #

    def add_listener(self, listener: Any):
        self._listeners.append(listener)

    # Note: There is a way to declare a useful type for the 'listener' parameter,
    # but it involves inheritance, which will be introduced in the
    # next project. For now I've just labeled it an Any, meaning anything.

    def notify_all(self, event_name: str, options={}):
        for listener in self._listeners:
            listener.notify(event_name, options=options)

    # ------ end of MVC ---------

    # (project part 2):  method 'approximate' goes here
    def approximate(self, from_index, to_index):
        self.notify_all("trial_approx", options={"p1": self.points[from_index], "p2": self.points[to_index]})
        self.approx.append(self.points[from_index])

        if to_index - from_index < 2:
            self.notify_all("final_approx_seg", options={"p1": self.points[from_index], "p2": self.points[to_index]})
            self.approx.append(self.points[from_index])

        else:
            dev_max = 0
            dev_point = 0
            for i in range(from_index + 1, to_index):
                dev = deviation(self.points[from_index], self.points[to_index], self.points[i])
                if dev > dev_max:
                    dev_max = dev
                    dev_point = i

            if dev_max <= self.tolerance:
                self.notify_all("final_approx_seg", options={"p1": self.points[from_index], "p2": self.points[to_index]})
                self.approx.append(self.points[from_index])
            else:
                self.approximate(from_index, dev_point)
                self.approximate(dev_point, to_index)

        
def deviation(p1: Point, p2: Point, p: Point) -> float:
    """Shortest distance from point p to a line through p1,p2"""
    intercept = normal_intercept(p1, p2, p)
    # Standard distance formula, sqrt((x2-x1)^2 +(y2-y1)^2)
    log.debug("Computing distance from {} to {}"
                  .format(p, (intercept.x,intercept.y)))
    dx = intercept.x - p.x
    dy = intercept.y - p.y
    return math.sqrt(dx*dx + dy*dy)

def normal_intercept(p1: Point, p2: Point, p: Point) -> Point:
    """
    The point at which a line through p1 and p2 
    intersects a normal dropped from p.  See normals.md
    for an illustration. 
    """
    log.debug("Normal intercept {}-{} from {}"
                  .format(p1, p2, p))

    # Special cases: slope or normal slope is undefined
    # for vertical or horizontal lines, but the intersections
    # are trivial for those cases
    if p2.x == p1.x:
        log.debug("Intercept at {}".format((p1.x,p.y)))
        return Point(p1.x, p.y)
    elif p2.y == p1.y:
        log.debug("Intercept at {}".format((p.x, p1.y)))
        return Point(p.x, p1.y)

    # The slope of the segment, and of a normal ray
    seg_slope = (p2.y - p1.y)/(p2.x - p1.x)
    normal_slope = 0 - (1.0 / seg_slope)

    # For y=mx+b form, we need to solve for b (y intercept)
    seg_b = p1.y - seg_slope * p1.x
    normal_b = p.y - normal_slope * p.x

    # Combining and subtracting the two line equations to solve for
    x_intersect = (seg_b - normal_b) / (normal_slope - seg_slope)
    y_intersect = seg_slope * x_intersect + seg_b
    # Colinear points are ok!

    log.debug("Intercept at {}".format(x_intersect, y_intersect))
    return Point(x_intersect, y_intersect)


    
    
