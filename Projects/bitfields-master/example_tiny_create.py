"""
Create a tiny 'program' encoded as binary numbers.
This example is for the bitfields project, in which we
have only the ALU and not the whole CPU and memory, so
the program just does a little arithmetic.
"""

from instr_format import instruction_from_string
from typing import List


def gen(s: str) -> int:
    """Create the integer encoding of one instruction from a string"""
    instr = instruction_from_string(s)
    encoding = instr.encode()
    return encoding


def make_sample() -> List[int]:
    """Create a small, arbitrary little sequence of
    instructions encoded as 32-bit binary integers.
    """
    return [
        gen("ADD  ALWAYS r1  r0 r0  15"),  # Initializes r1 to 15
        gen("ADD  ALWAYS r2  r1 r1  0"),   # r2 = r1 + r1
        gen("SUB  ALWAYS r3  r2 r0  5"),   # r3 = r2 - 5
        gen("MUL  ALWAYS r3  r3 r0  2")    # r3 = r3 * 2for
    ]


def main():
    """Store the tiny sample program in a file,
    which we'll call tiny.txt
    """
    outfile = open("tiny.txt", "w")
    for word in make_sample():
        print(word, file=outfile)
    outfile.close()


if __name__ == "__main__":
    main()
