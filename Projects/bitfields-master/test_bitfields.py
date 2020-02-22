"""
Tests for bitfield.py
"""

import unittest
import bitfield
from bitfield import BitField


class TestBitFields(unittest.TestCase):

    def test_low_order(self):
        low_4 = BitField(0, 3)
        # A value that fits snugly in the first 4 bits
        self.assertEqual(low_4.insert(13, 0), 13)
        # A value that doesn't fit; some high bits lost
        self.assertEqual(low_4.insert(21, 0), 5)
        # Extract unsigned
        self.assertEqual(low_4.extract(15), 15)
        # Or convert negative numbers 
        self.assertEqual(low_4.extract_signed(15), -1)
        # Doesn't clobber other bits
        higher = 15 << 4
        self.assertEqual(low_4.insert(13, higher), 13 + higher)
        # Extraction is masked
        packed = low_4.insert(13, higher)
        self.assertEqual(low_4.extract(packed), 13)

    def test_middle_bits(self):
        mid_4 = BitField(4, 7)
        # A value that fits snugly in4 bits
        self.assertEqual(mid_4.insert(13, 0), 13 << 4)
        # A value that doesn't fit; some high bits lost
        self.assertEqual(mid_4.insert(21, 0), 5 << 4)
        # Extract unsigned
        self.assertEqual(mid_4.extract(15 << 4), 15)
        # Or convert negative numbers 
        self.assertEqual(mid_4.extract_signed(15 << 4), -1)

    def test_invert(self):
        lowpart = BitField(0, 3)
        midpart = BitField(4, 6)
        highpart = BitField(7, 9)
        for v in range(8):
            packed = 0
            packed = lowpart.insert(v, packed)
            packed = midpart.insert(v, packed)
            packed = highpart.insert(v, packed)
            self.assertEqual(lowpart.extract(packed), v)
            self.assertEqual(midpart.extract(packed), v)
            self.assertEqual(highpart.extract(packed), v)

    def test_replace(self):
        """Test replacing an existing field of bits."""
        packed = 0x0f0  # 11110000  --- we want to clobber those ones
        newval = 0x005  # 00000101  --- and replaced them with 0101
        field = BitField(4, 7)  # ____xxxx  --- in positions 4..7
        packed = field.insert(newval, packed)
        self.assertEqual(packed, 0x50)  # 01010000 --- some zeros replaced 1s
        # Note 0x50 == 0b01010000 == 80

    def test_width(self):
        """Make sure we are masking out any bits that don't fit in 
        the field. 
        """
        second_nibble = BitField(4, 7)
        value = 0xff  # A full byte; two nibbles
        packed = second_nibble.insert(value, 0)
        self.assertEqual(packed, 0xf0)
        # Note 0xf0 == 240


class TestSignExtension(unittest.TestCase):
    """Testing the sign extension function.  If you move sign extension into
    the BitFields class, you may want to remove this test class.
    """

    def test_extend_neg(self):
        """0b111 in a 3-bit field is negative 1"""
        self.assertEqual(bitfield.sign_extend(7, 3), -1)

    def test_extend_neg(self):
        """0b0111 in a 4-bit field is positive 7"""
        self.assertEqual(bitfield.sign_extend(7, 4), 7)

    def test_not_all_neg(self):
        """For good measure, make sure it works for an integer that
        is not all 1s.
        """
        self.assertEqual(bitfield.sign_extend(11, 4), -5)
        self.assertEqual(bitfield.sign_extend(11, 5), 11)
        self.assertEqual(bitfield.sign_extend(13, 4), -3)
        self.assertEqual(bitfield.sign_extend(13, 5), 13)


if __name__ == '__main__':
    unittest.main()
