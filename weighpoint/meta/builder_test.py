from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from weighpoint.meta import builder


class MetaNetworkBuilderTest(tf.test.TestCase):
    def test_yield_flat_paths(self):
        x = {
            'c': {'d': [tf.constant(6)]},
            'a': [tf.constant(3), tf.constant(4)],
            'b': tf.constant(5)
        }
        self.assertEqual(
            tuple(builder.yield_flat_paths(x)),
            (('a', 0), ('a', 1), ('b',), ('c', 'd', 0)))
        AB = collections.namedtuple('AB', ['a', 'b'])
        x = [AB(3, 4), 2]
        self.assertEqual(
            tuple(builder.yield_flat_paths(x)),
            ((0, 'a'), (0, 'b'), (1,)))

    def test_batched_tensor(self):
        b = builder.MetaNetworkBuilder()
        with b:
            x = b.prebatch_input(shape=(4, 5), dtype=tf.float32)
            batched = builder.batched(x)
            self.assertEqual(batched.shape.as_list(), [None, 4, 5])
        b.preprocessor()

    def test_batched_ragged(self):
        b = builder.MetaNetworkBuilder()
        with b:
            x = b.prebatch_input(shape=(None, 5), dtype=tf.float32)
            batched = builder.batched(x)
            self.assertTrue(isinstance(batched, tf.RaggedTensor))
            self.assertEqual(batched.shape.as_list(), [None, None, 5])
        b.preprocessor()

    def _test_batched_ragged_ragged(self, row_split_vals, row_splits_size):
        from tensorflow.python.ops.ragged import ragged_tensor_shape as rts
        rts.broadcast_to
        b = builder.MetaNetworkBuilder()
        with b:
            values = b.prebatch_input(shape=(None, 5), dtype=tf.float32)
            # values = tf.keras.layers.Lambda(lambda x: 2*x)(values)
            row_splits = b.prebatch_input(
                shape=(row_splits_size,), dtype=tf.int64)
            x = tf.RaggedTensor.from_row_splits(values, row_splits)
            b.as_batched_model_input(x)
        b.preprocessor()

    def test_batched_ragged_ragged(self):
        self._test_batched_ragged_ragged(
            [[0, 2, 5], [0, 4, 6, 7]], None)
        self._test_batched_ragged_ragged(
            [[0, 2, 5], [0, 4, 6]], 3)


if __name__ == '__main__':
    tf.test.main()
