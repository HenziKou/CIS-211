"""
An LL parser (attempt) for CIS 211
"""

from compiler.lexer import TokenStream
from compiler import expr
from compiler.syntax import TokenCat
from typing import TextIO

import logging

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class InputError(Exception):
    """Raised when we can't parse the input"""
    pass


def parse(srcfile: TextIO) -> expr.Expr:
    """Interface function to LL parser of Dumbol"""
    stream = TokenStream(srcfile)
    return _program(stream)


#
# The grammar comes here.  It should follow this ebnf:
#
#  program ::=  block
#  block ::= { stmt }
#  stmt ::=  assign | loop | ifstmt
#  whilestmt ::= 'while' exp 'do' block 'od'
#  ifstmt ::= 'if' exp 'then' block ['else' block] 'fi'
#  assignment ::=  IDENT '=' exp
#  exp ::= term { ('+'|'-') term }
#  term ::= primary { ('*'|'/')  primary }
#  primary ::= IDENT | CONST | '(' exp ')'
#

# Predictions based on next token:
first = {}
first["ifstmt"] = {TokenCat.IF}
first["whilestmt"] = {TokenCat.WHILE}
first["assignment"] = {TokenCat.IDENT}
first["stmt"] = first["ifstmt"].union(first["whilestmt"], first["assignment"])
first["exp"] = {TokenCat.IDENT, TokenCat.CONST}  # Add LPAREN !


# Initial version: Just sums

def require(stream: TokenStream, category: TokenCat, desc: str = "", consume=False):
    """Requires the next token in the stream to match a specified category.
    Consumes and discards it if consume==True.
    """
    if stream.peek().kind != category:
        raise InputError(f"Expecting {desc or category}, but saw {stream.peek()} instead")
    if consume:
        stream.take()
    return


def _program(stream: TokenStream) -> expr.Expr:
    """
    program ::= block
    """
    left = _block(stream)
    require(stream, TokenCat.END)
    return left


def _block(stream: TokenStream) -> expr.Expr:
    """
    block ::= { stmt }
    """
    log.debug(f"Parsing block from token {stream.peek()}")
    if stream.peek().kind not in first["stmt"]:
        return expr.Pass()
    left = _stmt(stream)
    log.debug(f"Starting block with {left}")
    while stream.peek().kind in first["stmt"]:
        right = _stmt(stream)
        log.debug(f"Adding statement to block: {right}")
        left = expr.Seq(left, right)
    return left


def _stmt(stream: TokenStream) -> expr.Expr:
    """
    assignment ::= IDENT '=' expression ';'
    """
    if stream.peek().kind is TokenCat.WHILE:
        return _while(stream)
    if stream.peek().kind is TokenCat.IF:
        return _if(stream)
    if stream.peek().kind is not TokenCat.IDENT:
        raise InputError(f"Expecting identifier at beginning of assignment, got {stream.peek()}")
    target = expr.Var(stream.take().value)
    if stream.peek().kind is not TokenCat.ASSIGN:
        raise InputError("Expecting assignment symbol, got {stream.peek()}")
    stream.take()  # Discard token
    value = _expr(stream)
    if stream.peek().kind is not TokenCat.SEMI:
        raise InputError(f"Expecting semicolon after assignment, got {stream.peek()}")
    stream.take()  # Discard token
    return expr.Assign(target, value)


def _while(stream: TokenStream) -> expr.While:
    """
    whilestmt ::= 'while' exp 'do' block 'od'
    """
    require(stream, TokenCat.WHILE, consume=True)
    cond = _expr(stream)
    require(stream, TokenCat.DO, consume=True)
    block = _block(stream)
    require(stream, TokenCat.OD, consume=True)
    stmt = expr.While(cond, block)
    return stmt


def _if(stream: TokenStream) -> expr.If:
    require(stream, TokenCat.IF, consume=True)
    cond = _expr(stream)
    require(stream, TokenCat.THEN, consume=True)
    then_block = _block(stream)
    if stream.peek().kind == TokenCat.ELSE:
        require(stream, TokenCat.ELSE, consume=True)
        else_block = _block(stream)
        result = expr.If(cond, then_block, else_block)
    else:
        result = expr.If(cond, then_block, elsepart=expr.Pass())
    require(stream, TokenCat.FI, consume=True)
    return result


def _expr(stream: TokenStream) -> expr.Expr:
    """
    expr ::= term { ('+'|'-') term }
    """
    log.debug(f"parsing sum starting from token {stream.peek()}")
    left = _term(stream)
    log.debug(f"sum begins with {left}")
    while stream.peek().value in ["+", "-"]:
        op = stream.take()
        log.debug(f"expr addition op {op}")
        right = _term(stream)
        if op.value == "+":
            left = expr.Plus(left, right)
        elif op.value == "-":
            left = expr.Minus(left, right)
        else:
            raise InputError(f"What's that op? {op}")
    return left


def _term(stream: TokenStream) -> expr.Expr:
    """term ::= primary { ('*'|'/')  primary }"""
    left = _primary(stream)
    log.debug(f"term starts with {left}")
    while stream.peek().value in ["*", "/"]:
        op = stream.take()
        right = _primary(stream)
        if op.value == "*":
            left = expr.Times(left, right)
        elif op.value == "/":
            left = expr.Div(left, right)
        else:
            raise InputError(f"Expecting multiplicative op, got {op}")
    return left


def _primary(stream: TokenStream) -> expr.Expr:
    """Constants, Variables, and parenthesized expressions"""
    log.debug(f"Parsing primary with starting token {stream.peek()}")
    token = stream.take()
    if token.kind is TokenCat.CONST:
        log.debug(f"Returning Const node from token {token}")
        return expr.Const(int(token.value))
    elif token.kind is TokenCat.IDENT:
        log.debug(f"Variable {token.value}")
        return expr.Var(token.value)
    elif token.kind is TokenCat.LPAREN:
        nested = _expr(stream)
        require(stream, TokenCat.RPAREN, consume=True)
        return nested
    else:
        raise InputError(f"Confused about {token} in expression")
