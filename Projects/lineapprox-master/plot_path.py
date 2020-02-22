"""
Plot a list of points in a TkInter window.

"""

import graphics.graphics as graphics
import argparse
import view_simplify
import geometry

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def cli():
    """Command line arguments for plotting a CSV file of points"""
    parser = argparse.ArgumentParser("Plot from CSV")
    parser.add_argument(
        "csv", type=argparse.FileType('r'),  
        help="Path to CSV file of points")
    parser.add_argument(
        "width", type=int, default=300, 
        help="Window width, in pixels")
    parser.add_argument(
        "height", type=int, default=300,
        help="Window height, in pixels")
    parser.add_argument(
        "tolerance", type=int, default=1000,
        help="Error tolerance in meters")
    args = parser.parse_args()
    return args

def read_points(csv_file):
    """Each line in the CSV file should contain an 
    easting and a northing in columns 0 and 1.  The first
    row may be column headers, so we'll skip any non-numeric 
    entries.
    """
    points = [ ]
    for line in csv_file:
        fields = line.split(",")
        if len(fields) < 2:
            continue
        easting = fields[0]
        northing = fields[1]
        if not easting.isdecimal():
            continue
        points.append(geometry.Point(int(easting), int(northing)))
    return points


if __name__ == "__main__":
    args = cli()
    pts = read_points(args.csv)
    win = graphics.GraphWin("Sample", args.width, args.height)
    path = geometry.PolyLine()
    for pt in pts:
        path.append(pt)
    log.debug("Will plot in area 0..{}, 0..{}"
                  .format(args.width,args.height))
    view = view_simplify.View(win, path)
    view.plot("blue")
    input("Press enter to simplify")
    simpler = path.approximate(args.tolerance)
    simple_view = view_simplify.View(win, simpler)
    simple_view.plot("green")
    print("Simplified from {} points to {} points"
          .format(len(path._points), len(simpler._points)))
    input("Press enter to dismiss")
    
    

                          
    
