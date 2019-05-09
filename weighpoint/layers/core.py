from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from weighpoint.data import problems


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


@gin.configurable(module='layers')
def Dense(
        units, activation=None,
        kernel_regularizer=None, kernel_initializer=None,
        kernel_constraint=None, impl='base'):
    if impl == 'base':
        if kernel_initializer is None:
            kernel_initializer = 'glorot_uniform'
        return tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            kernel_constraint=kernel_constraint)
    elif impl.startswith('tfp_'):
        from weighpoint.layers import tfp as wtfp
        divergence_fn = wtfp.scaled_kl_divergence_fn(
            problems.get_current_problem().examples_per_epoch('train'))
        return wtfp.DenseProbabilistic(
            units, impl[4:],
            kernel_divergence_fn=divergence_fn,
            bias_divergence_fn=divergence_fn,
            activation=activation)
    else:
        raise ValueError('Unsupported impl "%s"' % impl)
