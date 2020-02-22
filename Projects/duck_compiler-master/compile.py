"""
Driver (main program) for expression compiler. 
Input is parsed by llparse.py to create an
Expr object.  The 'gen' methods in Expr walk over
the Expr tree and produce assembly code in the
Context object.
"""

from compiler.llparse import parse, InputError
from compiler.lexer import LexicalError
from compiler import codegen_context

import datetime
import argparse
import sys

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def cli() -> object:
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Expression Compiler")
    parser.add_argument("sourcefile", type=argparse.FileType('r'),
                        help="Source program text")
    parser.add_argument("outfile", type=argparse.FileType('w'),
                        nargs="?", default=sys.stdout,
                        help="Output file for assembly code")
    args = parser.parse_args()
    return args


def main():
    args = cli()
    context = codegen_context.Context()
    context.add_line("# Lovingly crafted by robots")
    context.add_line("# {} from {}".format(datetime.datetime.now(), args.sourcefile.name))
    context.add_line("#")
    # Memory mapped IO addressed hooked to special variables named 'in' and 'out'
    context.hook_var("in", 510)
    context.hook_var("out", 511)
    ok = True
    try:
        exp = parse(args.sourcefile)
        log.debug("Parsed to: {}".format(exp))
        work_register = context.alloc_reg()
        exp.gen(context, work_register)
        context.free_reg(work_register)
        context.add_line("\tHALT  r0,r0,r0")
        assm = context.get_lines()
        log.debug("assm = {}".format(assm))
        for line in assm:
            # noinspection PyUnresolvedReferences
            print(line, file=args.outfile)
        print("#Compilation complete")
    except InputError as e:
        print("Syntax error, bailing")
    except LexicalError as e:
        print("Lexical error, bailing")
    except Exception as e:
        print("Failed!")
        print(e)
        raise e


if __name__ == "__main__":
    main()
