from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
err = ImportError(
        'weighpoint.layers.tfp requires tensorflow_probability>=0.6 which is '
        'not install by default.')
try:
    import tensorflow_probability as tfp
except ImportError:
    raise err

if tfp.__version__.split('.')[1] < 0.6:
    raise err
del err


@gin.configurable
def scaled_kl_divergence_fn(examples_per_epoch, scale_factor=None):
    if examples_per_epoch is None:
        raise ValueError('`examples_per_epoch` cannot be `None`')
    examples_per_epoch = tf.cast(examples_per_epoch, tf.float32)
    scale_factor = (1. if scale_factor is None else scale_factor)
    scale_factor = scale_factor / examples_per_epoch

    def scaled_kl(q, p, ignore):
        return tfp.distributions.kl_divergence(q, p) * scale_factor
    return scaled_kl


_tfp_fns = {
    'flipout': tfp.layers.DenseFlipout,
    'local_reparameterization': tfp.layers.DenseLocalReparameterization,
    'reparameterization': tfp.layers.DenseReparameterization,
}


def DenseProbabilistic(units, impl='flipout', **kwargs):
    return _tfp_fns[impl](units, **kwargs)
