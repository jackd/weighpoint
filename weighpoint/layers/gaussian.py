from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from weighpoint.ops import gaussian as _g
from weighpoint.layers.utils import lambda_wrapper


continuous_truncated_gaussian = lambda_wrapper(
    _g.continuous_truncated_gaussian)


@gin.configurable
class ContinuousTruncatedGaussian(tf.keras.layers.Layer):
    def build(self, input_shapes):
        self._root_scale_factors = self.add_weight(
            'root_scale_factors', shape=(), dtype=self.dtype,
            initializer=tf.ones_initializer())
        self._scale_factor = tf.square(self._root_scale_factors)
        super(ContinuousTruncatedGaussian, self).build(input_shapes)

    @property
    def scale_factor(self):
        return self._scale_factor

    def call(self, inputs):
        from weighpoint.ops.gaussian import continuous_truncated_gaussian
        x_squared, max_radius_squared = inputs
        gaussian = continuous_truncated_gaussian(
            x_squared, max_radius_squared, self._scale_factor)
        return gaussian
