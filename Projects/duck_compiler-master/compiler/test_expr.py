"""
Tests for expr.py, 
  (excludes parsing)

"""

import unittest
from compiler import expr
from compiler.env import Env


class TestExpr(unittest.TestCase):

    def test_const(self):
        env = Env(expr.Const, expr.NO_VALUE)
        seven = expr.Const(7)
        self.assertEqual(repr(seven), 'Const(7)')
        self.assertEqual(str(seven), '7')
        self.assertEqual(seven.eval(env), expr.Const(7))

    def test_plus(self):
        env = Env(expr.Const, expr.NO_VALUE)
        eight = expr.Const(8)
        nine = expr.Const(9)
        x = expr.Var('x')
        y = expr.Var('y')
        self.assertEqual(expr.Plus(x, y), expr.Plus(x, y))
        self.assertEqual(expr.Plus(eight, nine).eval(env), expr.Const(17))
        self.assertEqual(expr.Plus(eight, expr.Plus(nine, eight)).eval(env),
                         expr.Const(25))
        self.assertEqual(repr(expr.Plus(eight, nine)),
                         "Plus(Const(8),Const(9))")
        self.assertEqual(str(expr.Plus(eight, nine)), "(8 + 9)")

    def test_minus(self):
        env = Env(expr.Const, expr.NO_VALUE)
        eight = expr.Const(8)
        nine = expr.Const(9)
        x = expr.Var('x')
        y = expr.Var('y')
        self.assertEqual(expr.Minus(x, y), expr.Minus(x, y))
        self.assertEqual(expr.Minus(nine, eight).eval(env), expr.Const(1))
        self.assertEqual(expr.Minus(eight, expr.Minus(nine, eight)).eval(env),
                         expr.Const(7))
        self.assertEqual(repr(expr.Minus(eight, nine)),
                         "Minus(Const(8),Const(9))")
        self.assertEqual(str(expr.Minus(eight, nine)), "(8 - 9)")

    def test_times(self):
        env = Env(expr.Const, expr.NO_VALUE)
        three = expr.Const(3)
        four = expr.Const(4)
        x = expr.Var('x')
        y = expr.Var('y')
        self.assertEqual(expr.Times(x, y), expr.Times(x, y))
        self.assertNotEqual(expr.Times(x, y), expr.Plus(x, y))
        self.assertEqual(expr.Times(three, four).eval(env), expr.Const(12))
        self.assertEqual(expr.Times(three, expr.Times(three, four)).eval(env),
                         expr.Const(36))

    def test_div(self):
        env = Env(expr.Const, expr.NO_VALUE)
        three = expr.Const(3)
        nine = expr.Const(9)
        x = expr.Var('x')
        y = expr.Var('y')
        self.assertEqual(expr.Div(x, y), expr.Div(x, y))
        self.assertNotEqual(expr.Div(x, y), expr.Times(x, y))
        self.assertEqual(expr.Div(nine, three).eval(env), expr.Const(3))
        self.assertEqual(expr.Div(nine, expr.Times(three, three)).eval(env),
                         expr.Const(1))

    def test_assign(self):
        """Assignments:  Assign(var, exp) """
        env = Env(expr.Const, expr.ZERO)
        x = expr.Var('x')
        y = expr.Var('y')
        # y is undefined here, should give value 0,
        # giving x value 9
        assignment = expr.Assign(x, expr.Plus(y, expr.Const(9)))
        # print(assignment)
        x_exp = assignment.eval(env)
        y_val = expr.Assign(y, expr.Const(3)).eval(env)
        # y == 3, x == 9 here
        result = expr.Plus(x, expr.Const(4)).eval(env)
        self.assertEqual(result, expr.Const(13))


if __name__ == '__main__':
    unittest.main()
