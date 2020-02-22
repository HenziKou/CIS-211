"""
A bit field is a range of binary digits within an
unsigned integer.   Bit 0 is the low-order bit,
with value 1 = 2^0.  Bit 31 is the high-order bit,
with value 2^31.

A bitfield object is an aid to encoding and decoding
instructions by packing and unpacking parts of the
instruction in different fields within individual
instruction words.

Note that we are treating Python integers as if they
were 32-bit unsigned integers.  They aren't ... Python
actually uses a variable length signed integer
representation, but we ignore that because we are trying
to simulate a machine-level representation.
"""

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

WORD_SIZE = 32

class BitField(object):
    """A BitField object handles insertion and
    extraction of one field within an integer.
    """

    def __init__(self, from_bit: int, to_bit: int) -> None:
        """Tool for inserting and extracting bits
        from_bit ... to_bit, where 0 is the low-order
        bit and 31 is the high-order bit of an unsigned
        32-bit integer. For example, the low-order 4 bits
        could be represented by from_bit=0, to_bit=3.
        """
        assert 0 <= from_bit < WORD_SIZE
        assert from_bit <= to_bit <= WORD_SIZE
        self.from_bit = from_bit
        self.to_bit = to_bit

        self.field_width = 1 + to_bit - from_bit

        # Masks for the field
        #   1s in the field position, 0s elsewhere
        self.mask = self._construct_mask(self.field_width) << from_bit
        #   0s in the field position, 1s elsewhere
        self.maskout = ~ self.mask

        # Sign extension complement and mask
        self.comp = 1 << self.field_width
        self.sign_bit = 1 << (self.field_width - 1)

    def _construct_mask(self, width: int) -> int:
        """Construct a mask of required width in the
        low-order bits.
        """
        assert width >= 0
        mask = 0
        for bit in range(width):
            mask = (mask << 1) + 1
        return mask

    def insert(self, field_value: int, word: int) -> int:
        """Insert value of field into word.
        For example,
          if word is   xaa00aa00 and
          field_val is x0000000f
          and the field is bits 4..7
        then insert gives xaa00aaf0
        """
        # The field value, shifted into place and masked
        in_place = field_value << self.from_bit
        in_place = in_place & self.mask
        # Clear bits from field in target value
        word &= self.maskout
        # Combine
        return word | in_place

    def extract(self, word: int) -> int:
        """Extract the bitfield and return it in the
        low-order bits.  For example, if we are extracting
        the high-order five bits, the result will be an
        integer between 0 and 31.
        """
        field = word & self.mask
        return field >> self.from_bit


    def extract_signed(self, word: int) -> int:
        """Extract the bitfield and return it in the
        low order bits, sign-extended.
        """
        unsigned = self.extract(word)
        # Sign extend if negative
        if unsigned & self.sign_bit:
            return unsigned - self.comp
        else:
            return unsigned


# Sign extension is a little bit wacky in Python, because Python
# doesn't really use 32-bit integers ... rather it uses a special
# variable-length bit-string format, which makes *most* logical
# operations work in the extpected way  *most* of the time, but
# with some differences that show up especially for negative
# numbers.  I've written this sign extension function for you so
# that you don't have to spend time plotting a way to make it work.
# You'll probably want to convert it to a method in the BitField
# class.
#
# Examples:
#    Suppose we have a 3 bit field, and the field
#    value is 0b111 (7 decimal).  Since the high
#    bit is 1, we should interpret it as
#    -2^2 + 2^1  + 2^0, or -4 + 3 = -1
#
#    Suppose we hve the same value, decimal 7 or
#    0b0111, but now it's in a 4 bit field.  In thata
#    case we should interpret it as 2^2 + 2^1 + 2^0,
#    or 4 + 2 + 1 = 7, a positive number.
#
#    Sign extension distinguishes these cases by checking
#    the "sign bit", the highest bit in the field.
#
def sign_extend(field: int, width: int) -> int:
    """Interpret field as a signed integer with width bits.
    If the sign bit is zero, it is positive.  If the sign bit
    is negative, the result is sign-extended to be a negative
    integer in Python.
    width must be 2 or greater. field must fit in width bits.
    """
    log.debug("Sign extending {} ({}) in field of {} bits".format(field, bin(field), width))
    assert width > 1
    assert field >= 0 and field < 1 << (width + 1)
    sign_bit = 1 << (width - 1) # will have form 1000... for width of field
    mask = sign_bit - 1         # will have form 0111... for width of field
    if (field & sign_bit):
        # It's negative; sign extend it
        log.debug("Complementing by subtracting 2^{}={}".format(width-1,sign_bit))
        extended = (field & mask) - sign_bit
        log.debug("Should return {} ({})".format(extended, bin(extended)))
        return extended
    else:
        return field










