"""
Basic (phase 1) tests for geometry.py,
Point and PolyLine.  These tests should pass
by the first phase of the project.
"""

from geometry import Point, PolyLine
import unittest

class PointTests(unittest.TestCase):
    """The Point class is a simple immutable container
    that contains an x and y coordinate.
    """

    def test_xy_fields(self):
        pt = Point(3,5)
        self.assertEqual(pt.x, 3)
        self.assertEqual(pt.y, 5)

    def test_xy_fields_float(self):
        pt = Point(0.7, -15.342)
        self.assertEqual(pt.x, 0.7)
        self.assertEqual(pt.y, -15.342)

    def test_point_equality(self):
        p1 = Point(5,8)
        p2 = Point(5,8)
        p3 = Point(5,9)
        self.assertEqual(p1,p2)
        self.assertNotEqual(p1,p3)

    def test_repr(self):
        p = Point(-15, 12)
        self.assertEqual(repr(p), "Point(-15, 12)")


class PolyLineBasic(unittest.TestCase):
    """Just the basic properties of PolyLine,
    before the simplification method is
    added.
    """

    def test_list_protocol(self):
        """A PolyLine should implement some of the same
        methods that the 'list' class implements, so that
        we can use it as if it were just a list of Points.
        """

        pl = PolyLine()
        self.assertEqual(len(pl), 0)
        pl.append(Point(0,0))
        self.assertEqual(len(pl), 1)
        self.assertEqual(pl[0], Point(0,0))
        pl.append(Point(2,3))
        self.assertEqual(len(pl), 2)
        self.assertEqual(pl[0], Point(0,0))
        self.assertEqual(pl[1], Point(2,3))
        self.assertEqual(pl[1],pl[-1])
        self.assertEqual(pl[0],pl[-2])
        with self.assertRaises(IndexError):
            p = pl[3]
        with self.assertRaises(IndexError):
            p = pl[-3]
        count = 0
        for pt in pl:
            self.assertEqual(pt, pl[count])
            count += 1


if __name__ == "__main__":
    unittest.main()

