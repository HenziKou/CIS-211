"""
Driver (main program) for intepreter.
Should process the same language as the
compiler, but interprets it rather than
compiling it into assembly code.
"""

from compiler.llparse import parse
from compiler import expr
from compiler.env import Env

import argparse
import sys

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def cli() -> object:
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Duck Language Interpreter")
    parser.add_argument("sourcefile", type=argparse.FileType('r'),
                        help="Source program text")
    parser.add_argument("outfile", type=argparse.FileType('w'),
                        nargs="?", default=sys.stdout,
                        help="Output file for assembly code")
    args = parser.parse_args()
    return args


def duck_in(varname: str) -> expr.Const:
    """Hook the 'in' variable as input"""
    while True:
        try:
            # The interpreter is more polite than the Duck Machine
            val = input("May I have an integer?")
            const = expr.Const(int(val))
            print("Thank you very much.")
            log.debug("Constructed input {}".format(const))
            return const
        except Exception as e:
            print("Well, that was not entirely satisfactory.")
            print("I got the following exception: {}".format(e))
            print("Would you mind very much trying again?")


def duck_out(val: expr.Const):
    """Hook the 'out' variable as output"""
    print("Program output: {}".format(val.value()))


def main():
    args = cli()
    try:
        exp = parse(args.sourcefile)
        log.debug("Parsed to: {}".format(exp))
        env = Env(expr.Const, expr.NO_VALUE)
        env.hook_input("in", duck_in)
        env.hook_output("out", duck_out)
        exp.eval(env)
        print("#Interpretation complete")
    except Exception as e:
        print("Failed!")
        print(e)
        raise e


if __name__ == "__main__":
    main()
