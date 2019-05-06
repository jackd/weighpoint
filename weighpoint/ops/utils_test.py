from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from weighpoint.ops import utils
from weighpoint.test_utils import RaggedTestCase


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(RaggedTestCase):

    def test_map_gather_ragged_indices(self):
        param_vals = np.random.uniform(size=(2, 10))
        params = tf.constant(param_vals[0], dtype=tf.float32)
        indices = tf.RaggedTensor.from_row_lengths(
            tf.constant([2, 3, 1, 5, 0], dtype=tf.int64),
            tf.constant([3, 2], dtype=tf.int64))

        out = tf.gather(params, indices)
        batched_params = tf.expand_dims(params, axis=0)
        batched_indices = tf.expand_dims(indices, axis=0)
        batched_out = utils.map_gather(
            batched_params, batched_indices)[0]

        self.assertRaggedEqual(*self.evaluate((out, batched_out)))

        params2 = tf.constant(param_vals[1], dtype=tf.float32)
        indices2 = tf.RaggedTensor.from_row_lengths(
            tf.constant([2, 6, 0, 2, 1], dtype=tf.int64),
            tf.constant([3, 2], dtype=tf.int64))
        out2 = tf.gather(params2, indices2)

        batched_params2 = tf.stack([params, params2], axis=0)
        batched_indices2 = tf.stack([indices, indices2], axis=0)
        batched_out2 = utils.map_gather(batched_params2, batched_indices2)
        out_stacked = tf.stack([out, out2], axis=0)

        self.assertRaggedEqual(*self.evaluate((out_stacked, batched_out2)))

    def test_map_gather_ragged_params(self):
        param_vals = np.random.uniform(size=(13, 10))
        param_vals = tf.constant(param_vals, dtype=tf.float32)
        params = tf.RaggedTensor.from_row_lengths(
            param_vals, tf.constant([4, 9], dtype=tf.int64))
        indices = tf.constant([
            [2, 3, 1, 1],
            [2, 5, 6, 8],
        ], dtype=tf.int64)
        actual = utils.map_gather(params, indices)
        expected = tf.stack([
            tf.gather(params[0], indices[0]),
            tf.gather(params[1], indices[1])
        ], axis=0)
        self.assertAllEqual(self.evaluate(actual), self.evaluate(expected))

    def test_flatten_final_dims(self):
        x = tf.constant(np.random.normal(size=(3, 4, 5, 6)))
        flat_x = self.evaluate(tf.reshape(x, (-1,)))
        y = utils.flatten_final_dims(x, n=2)
        self.assertAllEqual(y.shape.as_list(), [3, 4, 30])
        self.assertAllEqual(self.evaluate(tf.reshape(y, (-1,))), flat_x)
        y = utils.flatten_final_dims(x, n=3)
        self.assertAllEqual(y.shape.as_list(), [3, 120])
        self.assertAllEqual(self.evaluate(tf.reshape(y, (-1,))), flat_x)

    def test_flatten_leading_dims(self):
        x = tf.constant(np.random.normal(size=(3, 4, 5, 6)))
        flat_x = self.evaluate(tf.reshape(x, (-1,)))
        y = utils.flatten_leading_dims(x, n=2)
        self.assertAllEqual(y.shape.as_list(), [12, 5, 6])
        self.assertAllEqual(self.evaluate(tf.reshape(y, (-1,))), flat_x)
        y = utils.flatten_leading_dims(x, n=3)
        self.assertAllEqual(y.shape.as_list(), [60, 6])
        self.assertAllEqual(self.evaluate(tf.reshape(y, (-1,))), flat_x)

    def test_flatten_leading_dims_ragged(self):
        values = tf.constant(np.random.normal(size=(10, 10)))
        row_lengths = tf.constant([4, 6], dtype=tf.int64)
        x = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        flat_x = tf.reshape(values, (-1,))
        y = utils.flatten_leading_dims(x, n=2)
        self.assertAllEqual(y.shape.as_list(), [10, 10])
        self.assertAllEqual(self.evaluate(tf.reshape(y, (-1,))), flat_x)
        y = utils.flatten_leading_dims(x, n=3)
        self.assertAllEqual(y.shape.as_list(), [100])
        self.assertAllEqual(self.evaluate(y), flat_x)

    def test_reduce_sum(self):
        values = tf.constant(np.random.normal(size=(10, 10, 5)))
        row_lengths = tf.constant([2, 3, 5], dtype=tf.int64)
        x = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        for axis in (None, (1, 2), (-1,), (-2,)):
            s0 = tf.reduce_sum(x, axis)
            s1 = utils.reduce_sum(x, axis)
            self.assertRaggedEqual(*self.evaluate((s0, s1)))
        rl2 = tf.constant([1, 2], dtype=tf.int64)
        x2 = tf.RaggedTensor.from_row_lengths(x, rl2)
        for axis in (
                None, (-1,), 2, 3, 4, (3, 4), (2, 3, 4), (1, 2, 3, 4),
                (0, 1, 2, 3, 4)):
            s0 = tf.reduce_sum(x2, axis)
            s1 = utils.reduce_sum(x2, axis)
            self.assertRaggedEqual(*self.evaluate((s0, s1)))

    def test_diff(self):
        dims = (4, 5, 6)
        x = np.arange(np.prod(dims)).reshape(dims)
        axes = 0, 1, 2, -1, -2, -3
        ns = 1, 2
        for axis in axes:
            for n in ns:
                expected = np.diff(x, n=n, axis=axis)
                actual = self.evaluate(utils.diff(x, n=n, axis=axis))
                self.assertAllClose(actual, expected)


if __name__ == '__main__':
    tf.test.main()
