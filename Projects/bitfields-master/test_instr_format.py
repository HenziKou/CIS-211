"""
Very simple tests of the instruction format, encoding and decoding.
"""

import unittest
import instr_format
from instr_format import OpCode, CondFlag


class test_instruction(unittest.TestCase):
    """Simple shakedown of instruction format"""

    def test_instruction_from_string(self):
        """Simple smoke test: Can I make an instruction from a string?"""
        instr = instr_format.instruction_from_string("  ADD ALWAYS pc r3 r5 -12")
        self.assertEqual(instr.op, OpCode.ADD)
        self.assertEqual(instr.cond, CondFlag.ALWAYS)
        self.assertEqual(instr.reg_target, 15)
        self.assertEqual(instr.reg_src1, 3)
        self.assertEqual(instr.reg_src2, 5)
        self.assertEqual(instr.offset, -12)

    def test_instruction_from_dict(self):
        """Simple smoke test: Can I make an instruction from a dict?"""
        d = { "opcode": "MUL", "predicate": "ALWAYS", "target": "r1",
           "src1": "r2", "src2": "r3", "offset": "-12" }
        instr = instr_format.instruction_from_dict(d)
        self.assertEqual(instr.op, OpCode.MUL)
        self.assertEqual(instr.cond, CondFlag.ALWAYS)
        self.assertEqual(instr.reg_target, 1)
        self.assertEqual(instr.reg_src1, 2)
        self.assertEqual(instr.reg_src2, 3)
        self.assertEqual(instr.offset, -12)

    def test_encode_decode(self):
        """Binary encoding and decoding of an instruction"""
        instr = instr_format.instruction_from_string("  ADD ALWAYS pc r3 r5 -12")
        as_word = instr.encode()
        as_instr = instr_format.decode(as_word)
        self.assertEqual(instr, as_instr)


if __name__ == "__main__":
    unittest.main()

