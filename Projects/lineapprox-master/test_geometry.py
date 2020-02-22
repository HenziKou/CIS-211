"""
Tests for polyline simplification. 

"""

import unittest
import geometry
from geometry import Point, PolyLine
from typing import List, Tuple
import logging

def show_path_diff(actual: PolyLine, expected: PolyLine):
    """Debugging aid: Where did the path differ from what I expected?"""
    print("Diff ... ")
    if len(actual) == len(expected):
        print("(Lengths match)")
        minlen = len(actual)
    else:
        print(f"*** Lengths differ: {len(actual)} != {len(expected)}")
        minlen = min(len(actual, len(expected)))
    for i in range(minlen):
        print(f"{actual[i]}\t{expected[i]}")


class Test_Distance(unittest.TestCase):
    """Simple unit tests of distance calculation"""

    def test_from_floor(self):
        """Segment along horizontal axis"""
        self.assertEqual(
            geometry.deviation(Point(0,0),Point(100,0),Point(50,50)), 50)

    def test_from_level(self):
        """Segment from horizontal line, not at zero"""
        self.assertEqual(
            geometry.deviation(Point(100,100),Point(200,100),Point(150,300)), 200)

    def test_from_vertical(self):
        """Segment from vertical line"""
        self.assertEqual(
            geometry.deviation(Point(100,100),Point(100,200),Point(150,150)), 50)
        

def pontify(tuples: List[Tuple[int, int]]) -> List[Point]:
    """Convenience conversion"""
    return [ Point(x, y) for x,y in tuples]

def build(coords: List[Tuple[int, int]]) -> geometry.PolyLine:
    """Convenience function for building polylines"""
    path: PolyLine = geometry.PolyLine()
    for x, y in coords: 
        path.append(Point(x,y))
    return path
    

class TestSimplify(unittest.TestCase):
    """Simple unit tests for D-P polyline approximation"""

    def test_three_straight(self):
        """Should remove the intermediate point"""
        geometry.log.setLevel(logging.INFO)
        path = build([(0,0), (50,50), (100,100)])
        simplified = path.approximate(25)
        expected = build([(0,0), (100,100)])
        self.assertEqual(simplified, expected)

    def test_straight_simplify(self):
        """Should remove all intermediate points"""
        geometry.log.setLevel(logging.INFO)
        path = build([(0,0), (20,20), (30,30), (40,40), (50,50),
                      (60,60), (70,70), (80,80), (100,100) ])

        approx = path.approximate(25)
        expected = build([(0,0), (100,100)])
        self.assertEqual(approx, expected)

    def test_ridge(self):
        """Simpler than zigzag ... one ridge"""
        geometry.log.setLevel(logging.INFO)
        path = build([(0,0), (50,50), (100,100),
                  (150,50), (200,0)])
        approx = path.approximate(20)
        self.assertEqual(approx, build([(0,0), (100,100), (200, 0)]))

    def test_zigzag_simplify(self):
        geometry.log.setLevel(logging.INFO)    
        path = build([ (0,0), (50,100), (100,200), (150,100), (200,0),
                       (250,100), (300,200), (350,100), (400,0),
                       (450,100), (500,200), (550,100), (600,0) ])
        approx = path.approximate(75)
        expected = build([(0,0), (100,200), (200,0), (300,200), (400,0), (500,200), (600,0)])
        # show_path_diff(approx, expected)
        self.assertEqual(approx, expected)


if __name__ == "__main__":
    unittest.main()
