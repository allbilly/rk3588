"""Independent byte expectations for NHWC row padding; no NPU is required."""
import struct
import unittest

from tensor_io import pack, unpack


def half(values):
    return struct.pack(f"<{len(values)}e", *values)


class RowPadding(unittest.TestCase):
    def check_layout(self, shape, canonical, padded, stride, logical_format=1):
        elements = len(canonical)
        logical = {"dims": shape, "elements": elements, "format": logical_format}
        native = {"format": 1, "w_stride": stride,
                  "size": elements * 2, "size_with_stride": len(padded) * 2}
        self.assertEqual(pack(half(canonical), logical, native), half(padded))
        self.assertEqual(unpack(half(padded), logical, native), half(canonical))

    def test_two_rows(self):
        self.check_layout([1, 2, 2, 1], [1, 2, 3, 4],
                          [1, 2, 0, 0, 3, 4, 0, 0], 4)

    def test_batches_and_channels(self):
        self.check_layout([2, 2, 2, 3], list(range(1, 25)),
                          [1, 2, 3, 4, 5, 6, 0, 0, 0,
                           7, 8, 9, 10, 11, 12, 0, 0, 0,
                           13, 14, 15, 16, 17, 18, 0, 0, 0,
                           19, 20, 21, 22, 23, 24, 0, 0, 0], 3)

    def test_nchw_source(self):
        self.check_layout([1, 3, 2, 2], list(range(1, 13)),
                          [1, 5, 9, 2, 6, 10, 0, 0, 0,
                           3, 7, 11, 4, 8, 12, 0, 0, 0], 3, 0)

    def test_invalid_stride(self):
        logical = {"dims": [1, 2, 2, 1], "elements": 4, "format": 1}
        native = {"format": 1, "w_stride": 1, "size": 8, "size_with_stride": 8}
        with self.assertRaises(ValueError):
            pack(half([1, 2, 3, 4]), logical, native)
        with self.assertRaises(ValueError):
            unpack(half([1, 2, 3, 4]), logical, native)

    def test_single_pixel_nc1hwc2_padding(self):
        expected = half([1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0])
        for batch, channels, groups in [(2, 2, 1), (1, 4, 2)]:
            logical = {"dims": [batch, channels, 1, 1], "elements": 4, "format": 0}
            native = {"dims": [batch, groups, 1, 1, 2], "format": 2,
                      "w_stride": 3, "size": 8, "size_with_stride": 24}
            with self.subTest(batch=batch, channels=channels):
                self.assertEqual(pack(half([1, 2, 3, 4]), logical, native), expected)
                self.assertEqual(unpack(expected, logical, native), half([1, 2, 3, 4]))


if __name__ == "__main__":
    unittest.main()
