from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from weighpoint.np_utils.ragged_array import RaggedArray


ragged_lists = [[0, 1, 2, 3], [4, 5, 6], [7]]
flat_values = [0, 1, 2, 3, 4, 5, 6, 7]
row_lengths = [4, 3, 1]
row_splits = [0, 4, 7, 8]
leading_dim = 3
size = 8
shape = (3, None)

mask = np.array([False, True, True])
masked_values = [4, 5, 6, 7]
masked_row_lengths = [3, 1]
masked_row_splits = [0, 3, 4]


class RaggedArrayTest(unittest.TestCase):
    def _test(self, ragged):
        np.testing.assert_equal(ragged.ragged_lists, ragged_lists)
        np.testing.assert_equal(ragged.flat_values, flat_values)
        np.testing.assert_equal(ragged.row_lengths, row_lengths)
        np.testing.assert_equal(ragged.row_splits, row_splits)
        self.assertEqual(ragged.leading_dim, leading_dim)
        self.assertEqual(ragged.size, size)
        self.assertEqual(len(ragged), leading_dim)
        self.assertEqual(ragged.shape, shape)

    def test_from_row_lengths(self):
        self._test(RaggedArray.from_row_lengths(flat_values, row_lengths))

    def test_from_row_splits(self):
        self._test(RaggedArray.from_row_splits(flat_values, row_splits))

    def test_from_ragged_lists(self):
        self._test(RaggedArray.from_ragged_lists(ragged_lists))

    def test_indexing(self):
        ra = RaggedArray.from_row_splits(flat_values, row_splits)
        for i in range(len(ra)):
            np.testing.assert_equal(ragged_lists[i], ra[i])

    def test_iteration(self):
        ra = RaggedArray.from_row_splits(flat_values, row_splits)
        for rai, rli in zip(ra, ragged_lists):
            np.testing.assert_equal(rai, rli)

    def test_mask(self):
        ra = RaggedArray.from_row_splits(flat_values, row_splits)
        masked = ra.mask(mask)
        np.testing.assert_equal(masked.values, masked_values)
        np.testing.assert_equal(masked.row_splits, masked_row_splits)
        np.testing.assert_equal(masked.row_lengths, masked_row_lengths)


if __name__ == '__main__':
    unittest.main()
