"""
Interprets file tiny.txt as a simple
program.  Since we don't have the whole
CPU yet (that's next week), all we can do
is manipulate some values in "registers" (ints).

Expected output:
INFO:__main__:Executing ADD      r1,r0,r0[15]
INFO:__main__:Computed value 15
INFO:__main__:Executing ADD      r2,r1,r1[0]
INFO:__main__:Computed value 30
INFO:__main__:Executing SUB      r3,r2,r0[5]
INFO:__main__:Computed value 25
INFO:__main__:Executing MUL      r3,r3,r0[2]
INFO:__main__:Computed value 50
"""

from typing import List
from alu import ALU
import instr_format

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def execute(words: List[int]):
    """In place of a full CPU model, execute the ALU on a
    sequence of instructions encoded as integers.
    """
    # We don't have the Register objects yet, so ...
    regs = [0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]
    alu = ALU()
    for word in words:
        log.debug("Decoding instruction word {}".format(word))
        instr = instr_format.decode(word)
        log.debug("Decoded to {}".format(instr))
        op = instr.op
        to_reg = instr.reg_target
        src_1 = instr.reg_src1
        src_2 = instr.reg_src2
        offset = instr.offset
        log.info("Executing {}".format(instr))
        result, condition = alu.exec(op, regs[src_1], regs[src_2] + offset)
        regs[to_reg] = result
        log.info("Computed value {}".format(result))


def main():
    """Execute the tiny program created by
    example_tiny_create.py.
    """
    code = open("tiny.txt")
    words = []
    for line in code:
        words.append(int(line))
    execute(words)


if __name__ == "__main__":
    main()
