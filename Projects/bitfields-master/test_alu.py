"""
Simple tests for alu.py.  It should have an operation for each
defined opcode, and should return the appropriate condition code
(exactly one of N, P, or Z) along with the arithmetic result.
"""

import unittest
import alu
from instr_format import OpCode, CondFlag

"""  From instr_format.py: 
    HALT = 0    # Stop the computer simulation (in Duck Machine project)
    LOAD = 1    # Transfer from memory to register
    STORE = 2   # Transfer from register to memory
    # ALU operations
    ADD = 3     # Addition
    SUB = 5     # Subtraction
    MUL = 6     # Multiplication
    DIV = 7     # Integer division (like // in Python)
"""


class TestALU(unittest.TestCase):
    """Simple shakedown tests for alu.py"""

    def setUp(self):
        self.alu = alu.ALU()

    def test_halt(self):
        self.assertEqual(self.alu.exec(OpCode.HALT,42,64), (0,CondFlag.Z))

    def test_load(self):
        self.assertEqual(self.alu.exec(OpCode.LOAD,16,32), (48,CondFlag.P))

    def test_store(self):
        self.assertEqual(self.alu.exec(OpCode.STORE, 12, 18), (30, CondFlag.P))

    def test_add(self):
        self.assertEqual(self.alu.exec(OpCode.ADD, 19, 21), (40, CondFlag.P))
        self.assertEqual(self.alu.exec(OpCode.ADD, 19, -21), (-2, CondFlag.M))
        self.assertEqual(self.alu.exec(OpCode.ADD, 19, -19), (0, CondFlag.Z))

    def test_sub(self):
        self.assertEqual(self.alu.exec(OpCode.SUB, 19, 21), (-2, CondFlag.M))
        self.assertEqual(self.alu.exec(OpCode.SUB, 19, -21), (40, CondFlag.P))
        self.assertEqual(self.alu.exec(OpCode.SUB, -19, -19), (0, CondFlag.Z))

    def test_mul(self):
        self.assertEqual(self.alu.exec(OpCode.MUL, 4, 2), (8, CondFlag.P))
        self.assertEqual(self.alu.exec(OpCode.MUL, -4, -2), (8, CondFlag.P))
        self.assertEqual(self.alu.exec(OpCode.MUL, -4, 2), (-8, CondFlag.M))
        self.assertEqual(self.alu.exec(OpCode.MUL, 0, -19), (0, CondFlag.Z))

    def test_div(self):
        self.assertEqual(self.alu.exec(OpCode.DIV, 4, 2), (2, CondFlag.P))
        self.assertEqual(self.alu.exec(OpCode.DIV, -4, -2), (2, CondFlag.P))
        self.assertEqual(self.alu.exec(OpCode.DIV, -4, 2), (-2, CondFlag.M))
        self.assertEqual(self.alu.exec(OpCode.DIV, 0, 4), (0, CondFlag.Z))
        self.assertEqual(self.alu.exec(OpCode.DIV, 4, 0), (0, CondFlag.V))


if __name__ == "__main__":
    unittest.main()
