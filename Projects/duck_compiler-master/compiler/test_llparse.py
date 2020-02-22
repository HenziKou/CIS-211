"""Shakedown test of LL parser"""

from compiler import llparse
import io

sample = io.StringIO("""
 if 24 - 3 then 
      x = 17 + y ; 
      y = 34 * x + 4 ; 
 fi 
 while 14 do 
    x = 21 ; 
    z = ( 5 + 3 ) * 4 ; 
  od
""")
# StringIO is not the type expected by
# llparse.parse, but it is close enough to work
# for testing.
# noinspection PyTypeChecker
e = llparse.parse(sample)
print(e)
