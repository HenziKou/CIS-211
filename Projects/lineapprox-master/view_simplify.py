"""
View (animate) line simplification using the 
Douglas-Peucker (aka Ramer-Douglas-Peucker) 
algorithm. Implemented as a view object so that 
we can keep the canvas and coordinate transforms
as object state to be applied consistently. 
"""

import graphics.graphics as graphics
import geometry
from geometry import Point
from typing import List, Tuple

import logging
logging.basicConfig()
log = logging.getLogger(__name__)

class View(object):
    """A view of line simplification"""

    def __init__(self, win: graphics.GraphWin,
                     path: geometry.PolyLine, margin=20):
        self.win = win
        self.path = path
        path.add_listener(self)
        from_ll, from_ur = bbox(path)
        # Map easting,northing coordinates into window coordinates,
        #   scaling x and y coordinates uniformly and flipping the
        #   y axis so that north = up 
        self.tx = Transform(
            from_ll = from_ll,
            from_ur = from_ur,
            to_ll = Point(margin, margin),
            to_ur = Point(win.width - margin, win.height - margin),
            uniform=True, y_flip=True
            )

    def plot(self, color="blue"):
        screen_points = [(pt.x, pt.y)
                             for pt in self.tx.transform(self.path)]
        self.view_original = graphics.PolyLine(screen_points)
        self.view_original.setOutline(color)
        self.view_original.draw(self.win)


    def notify(self, event_name, options):
        """Event notifications from simplification process.
        :rtype: object
        """
        if event_name == "trial_approx":
            p1 = options["p1"]
            p2 = options["p2"]
            self.draw_segment(p1, p2, color="grey")
        elif event_name == "final_approx_seg":
            p1 = options["p1"]
            p2 = options["p2"]
            self.draw_segment(p1, p2, color="red")
        else:
            raise Exception("Unknown event {}".format(event_name))

    def draw_segment(self, p1, p2, color="green"):
        """Draw segment between two points. Returns an identifier
        that can be used to erase that segment later.
        """
        p1_screen = self.tx.transform_pt(p1)
        p2_screen = self.tx.transform_pt(p2)
        seg = graphics.Line(graphics.Point(p1_screen.x, p1_screen.y),
                                graphics.Point(p2_screen.x, p2_screen.y))
        seg.setOutline(color)
        seg.draw(self.win)
        return seg

def bbox(points: List[Point]) -> Tuple[Point, Point]:
    """Find the bounding box of a set of points.
    Typically used to find the range of model coordinates
    for initializing a Transform.
    Points is a sequence of Point objects (at least 1).
    Return the lower left and upper right corners of
    the bounding box.
    """
    assert len(points) > 0
    x_coords = [pt.x for pt in points]
    y_coords = [pt.y for pt in points]
    return Point(min(x_coords), min(y_coords)), Point(max(x_coords), max(y_coords))



class Transform:
    """A Transform translates and scales a set of
    points into a new range.  Typically used to
    transform "world" coordinates (e.g., kilometers)
    into "screen" coordinates (pixels).
    """

    def __init__(self, uniform=True, y_flip=True,
                 to_ll=Point(0, 0), to_ur=Point(100, 100),
                 from_ll=Point(0, 0), from_ur=Point(100, 100)):
        """Create a transformation from model coordinates
        with x and y ranging from from_ll to from_ur
        to window coordinates to_ll to to_ur.  y_flip
        indicates that model coordinate y axis goes up,
        window coordinate y axis goes down.
        """

        if y_flip:
            # Reverse y axis
            to_ll.y, to_ur.y = to_ur.y, to_ll.y

        sfx = (to_ur.x - to_ll.x) / (from_ur.x - from_ll.x)
        sfy = (to_ur.y - to_ll.y) / (from_ur.y - from_ll.y)

        log.debug("Scale factor x {}, Scale factor y {}".format(sfx, sfy))

        if uniform and y_flip:
            sfx = min(sfx, -sfy)
            sfy = max(-sfx, sfy)
        elif uniform:
            sfx = min(sfx, sfy)
            sfy = min(sfx, sfy)
            log.debug("Adjusted scale factor x {}, Scale factor y {}"
                      .format(sfx, sfy))

        # Scale factors
        self.sfx = sfx
        self.sfy = sfy
        # Translation
        self.to_ll = to_ll
        self.from_ll = from_ll

    def transform(self, points: List[Point]) -> List[Point]:
        """Apply transform to a list of points, or to
        a single point (x,y).
        """
        return [ self.transform_pt(pt) for pt in points]

    def transform_pt(self, pt: Point) -> Point:
        """Apply transformation to a single point"""

        x_scaled = self.to_ll.x + self.sfx * (pt.x - self.from_ll.x)
        y_scaled = self.to_ll.y + self.sfy * (pt.y - self.from_ll.y)
        return Point(x_scaled, y_scaled)






