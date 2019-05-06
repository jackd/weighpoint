from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from weighpoint import visitor


class RaggedTestCase(tf.test.TestCase):
    def evaluate(self, structure):
        structure = visitor.ragged_to_components(structure)
        structure = super(RaggedTestCase, self).evaluate(structure)
        structure = visitor.components_to_numpy(structure)
        return structure

    def assertRaggedEqual(self, a, b):
        if hasattr(a, 'flat_values'):
            self.assertTrue(hasattr(b, 'flat_values'))
            self.assertAllClose(a.flat_values, b.flat_values)
            self.assertEqual(
                len(a.nested_row_splits), len(b.nested_row_splits))
            for ar, br in zip(a.nested_row_splits, b.nested_row_splits):
                self.assertAllEqual(ar, br)
        else:
            self.assertAllClose(a, b)
