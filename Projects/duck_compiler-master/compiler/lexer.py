"""
Lexical analysis to convert input file into
streams of tokens.  Input string must delimit tokens
by spaces.  

Based on lexer.py from symbolic calculator project, 
but modified to read from a file. 

Author: Michal Young (michal@cs.uoregon.edu), March 2018
"""
from typing import Sequence, TextIO

import logging

from compiler import syntax

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# The operation symbols we recognize here are
# based on file syntax.py
OPSYMS = syntax.OPS.keys()


class LexicalError(Exception):
    """Raised when we can't extract tokens from the input"""
    pass


class Token(object):
    """One token from the input stream"""

    def __init__(self, value: any, kind: syntax.TokenCat):
        self.value = value
        self.kind = kind

    def __repr__(self) -> str:
        return "Token({}: {})".format(repr(self.value), self.kind)

    def __str__(self) -> str:
        return repr(self)


END = Token("End of Input", syntax.TokenCat.END)


class TokenStream(object):
    """
    Provides the tokens within a string one-by-one.
    """

    def __init__(self, f: TextIO):
        self.file = f
        self.tokens = []
        self._check_fill()
        log.debug("Tokens: {}".format(self.tokens))

    def __str__(self) -> str:
        return "[{}]".format("|".join(self.tokens))

    def _check_fill(self):
        while len(self.tokens) == 0:
            # We could read more than one line before hitting
            # a token, but the loop will be broken if we
            # hit end of file
            line = self.file.readline()
            if len(line) == 0:
                # End of file, leave zero tokens in buffer
                break
            self.tokens = lex(line.strip())
            log.debug("Refilled, tokens: {}".format(self.tokens))
            # Note this might also leave zero tokens in buffer,
            # but in that case outer while loop will attempt
            # to refill it until we either get some tokens
            # or hit end of file

    def has_more(self) -> bool:
        """True if there are more tokens in the stream"""
        self._check_fill()
        return len(self.tokens) > 0

    def peek(self) -> Token:
        """Examine next token without consuming it. """
        self._check_fill()
        if len(self.tokens) > 0:
            token = self.tokens[0]
        else:
            token = END
        return token

    def take(self) -> Token:
        """Consume next token"""
        self._check_fill()
        if len(self.tokens) > 0:
            token = self.tokens.pop(0)
        else:
            token = END
        return token


def lex(s: str) -> Sequence[Token]:
    """Break string into a list of Token objects"""
    words = s.split()
    tokens = []
    for word in words:
        if word.startswith("#"):
            break  # Discard comments
        tokens.append(classify(word))
    return tokens


def classify(word: str) -> Token:
    """Convert a textual token into a Token object
    with a value and category.
    """
    for kind in syntax.TokenCat:
        log.debug(f"Checking {word} for token class {kind}")
        pattern = kind.value
        if pattern.fullmatch(word):
            log.debug(f"Classified as {kind}")
            return Token(word, kind)
    raise LexicalError("Unrecognized token '{}'".format(word))
