"""
Associate syntactic elements like "*" recognized
by the parser with semantic classes like TIMES in 
the expr module.

Initially, for our postfix (RPN) parser for the
calculator, I did not try to associate tokens with
regular expressions.  However, as the language for
the compiler has grown, building patterns into the
list of tokens seems worthwhile.  However, to keep it simple,
I'll still use white space to separate tokens.
"""

from compiler import expr
from enum import Enum
import re


# Token categories.
class TokenCat(Enum):
        WHILE = re.compile("while")
        DO = re.compile("do")
        OD = re.compile("od")
        IF = re.compile("if")
        THEN = re.compile("then")
        ELSE = re.compile("else")
        FI = re.compile("fi")
        ASSIGN = re.compile("=")
        SEMI = re.compile(";")
        IDENT = re.compile(r"[a-zA-Z]\w*")
        MULOP = re.compile(r"[*/]")
        ADDOP = re.compile(r"[-+]")
        UNOP = re.compile(r"~")
        CONST = re.compile(r"[0-9]+")
        LPAREN = re.compile(r"\(")
        RPAREN = re.compile(r"\)")
        END = re.compile("--- EOF ---")  # should not match anything, but type-compatible


# Category names no longer used ...
# use the TokenCat enum instead.

# Expression nodes should be bound to a
# symbol and class here (excluding CONST and IDENT).
# We use this only for expression parsing; larger syntactic
# constructs have custom code.
# noinspection PyPep8
OPS = { "*": (TokenCat.MULOP, expr.Times)
        , "+": (TokenCat.ADDOP, expr.Plus)
        , "-": (TokenCat.ADDOP, expr.Minus)
        , "/": (TokenCat.MULOP, expr.Div)
        , "=": (TokenCat.ASSIGN, expr.Assign)  # Later we'll want separate ==, <=, etc.
        , "~": (TokenCat.UNOP, expr.Neg)
      }
