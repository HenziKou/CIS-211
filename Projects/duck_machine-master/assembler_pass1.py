"""
Assembler for DM2018W assembly language.

This assembler is for fully resolved instructions,
which may be the output of assm_xform.py, which
transforms instructions with symbolic addresses into
instructions with fully resolved (PC-relative) addresses.

Assembly instruction format with all options is

label: instruction

Labels are resolved (translated into addresses) in
assm_xform.py; in this pass of the interpreter they
are only for documentation.

Both parts are optional:  A label may appear without
an instruction, and an instruction may appear without
a label.

A label is just an alphabetic string, eg.,
  myDogBoo but not Betcha_5_Dollars

An instruction has the following form:

  opcode/predicate  target,src1,src2[disp]

Opcode is required, and should be one of the DM2018W
instruction codes (ADD, MOVE, etc); case-insensitive

/predicate is optional.  If present, it should be some
combination of N,Z,P, e.g., /NP would be "execute if
not zero".  If /predicate is not given, it is interpreted
as /ALWAYS, which is an alias for /NZP.

target is a register number (r0,r1, ... r15) or one of the
register aliases ZERO, PC, SP, etc.

src1 and src2 are likewise register specifiers.

[disp] is optional.  If present, it is a 12 bit
signed integer displacement.  If absent, it is
treated as [0].

DATA is a pseudo-operation:
   myvar:  DATA   18
indicates that the integer value 18
should be stored at this location, rather than
a DM2018W instruction.


Translate assembly language into assembly code
assembler_pass2.py translates assembly code into object code

assembly language  --Part 1-->  assembly code  --Part 2-->  object code

Author: Henzi Kou
"""

import argparse

from typing import List, Tuple
from enum import Enum, auto

import sys
import re
import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Configuration constants
ERROR_LIMIT = 5  # Abandon assembly if we exceed this


# Exceptions raised by this module
class SyntaxError(Exception):
    pass


###
# The whole instruction line is encoded as a single
# regex with capture names for the parts we might
# refer to. Error messages will be crappy (we'll only
# know that the pattern didn't match, and not why), but
# we get a very simple match/process cycle.  By creating
# a dict containing the captured fields, we can determine
# which optional parts are present (e.g., there could be
# label without an instruction or an instruction without
# a label).
###


# To simplify client code, we'd like to return a dict with
# the right fields even if the line is syntactically incorrect.
DICT_NO_MATCH = {'label': None, 'opcode': None, 'predicate': None,
                 'target': None, 'src1': None, 'src2': None,
                 'offset': None, 'comment': None}


###
# Although the DM2018W instruction set is very simple, a source
# line can still come in several forms.  Each form (even comments)
# can start with a label.
###

class AsmSrcKind(Enum):
    """Distinguish which kind of assembly language instruction
    we have matched.  Each element of the enum corresponds to
    one of the regular expressions below.
    """
    # Blank or just a comment, optionally
    # with a label
    COMMENT = auto()
    # Fully specified  (all addresses resolved)
    FULL = auto()
    # A data location, not an instruction
    DATA = auto()
    # Add the symbol choice for the new kind of instruction
    SYMBOLIC = auto()


# Lines that contain only a comment (and possibly a label).
# This includes blank lines and labels on a line by themselves.
#
ASM_COMMENT_PAT = re.compile(r"""
    # Optional label 
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   \s*
   # Optional comment follows # or ; 
   (
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)

# Instructions with fully specified fields. We can generate
# code directly from these.  In the transformation phase we
# pass these through unchanged, just keeping track of how much
# room they require in the final object code.
ASM_FULL_PAT = re.compile(r"""
   # Optional label 
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   # The instruction proper 
   \s*
    (?P<opcode>    [a-zA-Z]+)           # Opcode
    (/ (?P<predicate> [a-zA-Z]+) )?     # Predicate (optional)
    \s+
    (?P<target>    r[0-9]+),            # Target register
    (?P<src1>      r[0-9]+),            # Source register 1
    (?P<src2>      r[0-9]+)             # Source register 2
    (\[ (?P<offset>[-]?[0-9]+) \])?     # Offset (optional)
   # Optional comment follows # or ; 
   (
     \s*
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)

# Defaults for values that ASM_FULL_PAT makes optional
INSTR_DEFAULTS = [('predicate', 'ALWAYS'), ('offset', '0')]

# A data word in memory; not a DM2018W instruction
#
ASM_DATA_PAT = re.compile(r""" 
   # Optional label 
   (
     (?P<label> [a-zA-Z]\w*):
   )?
   # The instruction proper  
   \s*
    (?P<opcode>    DATA)           # Opcode
   # Optional data value
   \s*
   (?P<value>  (0x[a-fA-F0-9]+)
             | ([0-9]+))?
    # Optional comment follows # or ; 
   (
     \s*
     (?P<comment>[\#;].*)
   )?       
   \s*$             
   """, re.VERBOSE)

# Assembly code for ENUM

ASM_SYMBOLIC_PAT = re.compile(r"""
    # Optional label
    (
      (?P<label>   [a-zA-Z]\w*):
    )?
    # The instruction proper
    \s*
      (?P<opcode>    (JUMP)|(STORE)|(LOAD))           # Opcode
      (/ (?P<predicate> [a-zA-Z]+) )?                 # Predicate (optional)
    \s+
    ((?P<target>    r[0-9]+),)? 
    (?P<symbol>    [a-zA-Z]\w*)
    # Optional comment follows # or ; 
   (
     \s*
     (?P<comment>[\#;].*)
   )?       
   \s*$         
    """, re.VERBOSE)

PATTERNS = [(ASM_FULL_PAT, AsmSrcKind.FULL),
            (ASM_DATA_PAT, AsmSrcKind.DATA),
            (ASM_COMMENT_PAT, AsmSrcKind.COMMENT),
            (ASM_SYMBOLIC_PAT, AsmSrcKind.SYMBOLIC)
            ]


def parse_line(line: str) -> dict:
    """Parse one line of assembly code.
    Returns a dict containing the matched fields,
    some of which may be empty.  Raises SyntaxError
    if the line does not match assembly language
    syntax. Sets the 'kind' field to indicate
    which of the patterns was matched.
    """
    log.debug("\nParsing assembler line: '{}'".format(line))
    # Try each kind of pattern
    for pattern, kind in PATTERNS:
        match = pattern.fullmatch(line)
        if match:
            fields = match.groupdict()
            fields["kind"] = kind
            log.debug("Extracted fields {}".format(fields))
            return fields
    raise SyntaxError("Assembler syntax error in {}".format(line))


def build_table(lines: List[str]) -> Tuple[dict, int]:
    """
    Creates a symbol table that returns a dict. Your new method should take that list of lines as an input,
    and should return a dict that associates addresses with labels.

    Pseudocode:
        First Pass
        Build table - dictionary that maps labels to addresses
        Go through each line
        if the line has a label, then put the label and its corresponding address in the dict
        if it is not a comment-type line (use regex to figure this out), then increment our address
        The Tuple that is returned will include a count for the number of syntax errors, key errors, and exceptions.
    """

    error_count = 0
    address = 0
    symbol_table = {}
    for lnum in range(len(lines)):
        line = lines[lnum]
        log.debug("Pass 1 line {} address {}: {}".format(lnum, address, line))
        try:
            fields = parse_line(line)
            # Add/change 'if', 'elif', or 'else' statements belows
            if fields["label"]:
                lab = fields["label"]
                if lab in symbol_table:
                    print("Duplicate label {} on line {}".format(lab, lnum))
                    error_count += 1
                else:
                    symbol_table[lab] = address
            if fields["kind"] != AsmSrcKind.COMMENT:
                address += 1
        except SyntaxError as e:
            error_count += 1
            print("Syntax error in line {}: {}".format(lnum, line))
        except KeyError as e:
            error_count += 1
            print("Unknown word in line {}: {}".format(lnum, e))
        except Exception as e:
            error_count += 1
            print("Exception encountered in line {}: {}".format(lnum, e))
        if error_count > ERROR_LIMIT:
            print("Too many errors; abandoning")
            sys.exit(1)
    return symbol_table, error_count


def transform_lines(lines: List[str], symbol_table: dict) -> None:
    """
    Goes through each line and asks does this line have a sym.pat if so, call resolve_lines.
    It doesn't return anything, but rather makes modifications to the list of lines.
    It only changes lines that match the new pattern.

    Pseudocode:
        Second Pass
        going through all the lines again
        in order to calculate the "pc-relative-address" for resolving - need to know:
            where we are in our program
            (i.e. another address counter will show up here)
    """

    address = 0
    for lnum in range(len(lines)):
        line = lines[lnum]
        log.debug("Pass 2 line {}, address {}: {}".format(lnum, address, line))
        fields = parse_line(line)
        if fields["kind"] == AsmSrcKind.SYMBOLIC:
            try:
                lines[lnum] = resolve_line(fields, address, symbol_table)
            except KeyError:
                print("Unresolved symbol: {}".format(fields["symbol"]))
        if fields["kind"] != AsmSrcKind.COMMENT:
            address += 1


def resolve_line(fields: dict, addr: int, symtab: dict) -> str:
    """
    Changes JUMP, STORE, and LOAD with arguments: fields, address, and symbol table.

    Pseudocode:
        used by transform_lines function
        take a dictionary of fields -> correctly formatted string
        use the .format method a lot here
        will need several cases, one for each load, store, jump
    """

    op = fields["opcode"]
    label = fields["label"]
    pre = fields["predicate"]
    tar = fields["target"]
    sym = fields["symbol"]
    distance = symtab[sym] - addr

    # Check to see if symbol is in the symbol table created
    if fields["symbol"] not in symtab:
        raise SyntaxError("Symbol does not exist in table")
    if pre:
        predicate = "/{}".format(pre)
    else:
        predicate = ""

    # Check if there is a label
    if label is None:
        lab = ""
    else:
        lab = "{}: ".format(label)

    # Check the three different opcodes: LOAD, STORE, and JUMP
    if op == "STORE" or op == "LOAD":
        tar = fields["target"]
        return "{} {} {},r0,r15[{}] # Access variable '{}'".format(lab, op, tar, distance, sym)
    elif op == "JUMP":
        return "{} ADD{} r15,r0,r15[{}] #Jump to {}".format(lab, predicate, distance, sym)


def cli() -> object:
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Duck Machine Assembler (pass 2)")
    parser.add_argument("sourcefile", type=argparse.FileType('r'),
                        nargs="?", default=sys.stdin,
                        help="Duck Machine assembly code file")
    # Change parser argument below
    parser.add_argument("outfile", type=argparse.FileType('w'),
                        nargs="?", default=sys.stdout,
                        help="Output file for resolved assembly code")
    args = parser.parse_args()
    return args


def main():
    """"Assemble a Duck Machine program"""
    args = cli()
    source_lines = [line.rstrip() for line in args.sourcefile.readlines()]
    symtab, errors = build_table(source_lines)
    log.debug("Symbol table: {}".format(symtab))
    if errors == 0:
        transform_lines(source_lines, symtab)
        for line in source_lines:
            print(line, file=args.outfile)
    args.outfile.close()

    log.debug("Done")

if __name__ == "__main__":
    main()





