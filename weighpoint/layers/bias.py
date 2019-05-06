from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class AddBias(tf.keras.layers.Layer):
    def __init__(self, initializer=None, dtype=tf.float32, **kwargs):
        if initializer is None:
            initializer = tf.keras.initializers.Zeros()
        self._initializer = initializer
        super(AddBias, self).__init__(dtype=dtype, **kwargs)

    def build(self, input_shape):
        self._bias = self.add_weight(
            'bias', shape=(input_shape[-1],), initializer=self._initializer)
        super(AddBias, self).build(input_shape)

    def call(self, inputs):
        return inputs + self._bias


def add_bias(tensor, **kwargs):
    return AddBias(**kwargs)(tensor)
