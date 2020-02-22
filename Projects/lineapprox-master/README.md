# README #

Apply Douglas-Peucker line simplification algorithm and display 
the before and after version. Python 3. 

This is intended to be a programming exercise near the beginning of 
CIS 211, which spans the end of ACM CS1 and ACM CS2 in the ACM 
curriculum.   It is partly an introduction to classes and objects (without inheritance), and partly an exercise in  divide-and-conquer 
problem solving using recursion.  For many students it may also be a first experience working on a project that involves several source files, although they only need to make changes to one. 

## Manifest

* geometry.py:  Defines the PolyLine data structure with operations
that include Douglas-Peucker simplification to a given tolerance
* test_geometry.py:  Unit tests for geometry.py
* view_simplify.py:  Visualization of a path and animation of
approximation algorithm
* plot_path.py:  Driver program --- Plot a geometry (.csv) file and
illustrate its simplification.
* plot_path_basic.py:  Driver program for just part 1 of the project
* test_geometry_basic.py=: Test suite for just part 1 of the project
* graphics:  This subdirectory holds a slightly modified version of Zelle's Python graphics package, which uses TkInter.  

These files follow a Model-View-Controller scheme.  We will discuss that in week 2. 

## Project requirements

See [https://classes.cs.uoregon.edu/18S/cis211/projects/lineapprox.php](https://classes.cs.uoregon.edu/18S/cis211/projects/lineapprox.php)

## To test

```python3 test_geometry.py```

The student starter code should fail several tests until the the simplify method has been completed.

For part 1 of the project, use 

```python3 test_geometry_basic.py```
 

## To demonstrate

```python3 plot_path.py data/FoxHollow.csv 600 600 50```

This will simplify a 17 mile path around Eugene from 1502 points
to 61 points while deviating no more than 50 meters from the full
path. Expected output (in addition to graphical display): 

```
Press enter to simplify
Simplified from 1502 points to 61 points
Press enter to dismiss
```


### Who do I talk to? ###

* Developed by Michal Young, michal@cs.uoregon.edu
