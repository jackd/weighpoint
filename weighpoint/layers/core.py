from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


@gin.configurable(blacklist=['x'])
def relu_batch_norm(x, training=None, scale=True):
    x = tf.keras.layers.Activation('relu')(x)
    return tf.keras.layers.BatchNormalization(scale=scale)(
        x, training=training)


@gin.configurable(blacklist=['x'])
def batch_norm(x, training=None, scale=True):
    return tf.keras.layers.BatchNormalization(scale=scale)(
        x, training=training)


@gin.configurable(blacklist=['x'])
def dropout(x, rate=0.5, training=None):
    if rate is None:
        return x
    return tf.keras.layers.Dropout(rate)(x, training=training)


Dense = gin.config.external_configurable(
        tf.keras.layers.Dense, module='layers')
