from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gin
import six
import tensorflow as tf
from weighpoint.layers import conv
from weighpoint.layers import core
from weighpoint.layers import utils


class Convolver(object):
    @abc.abstractmethod
    def in_place_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        raise NotImplementedError

    @abc.abstractmethod
    def resample_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        raise NotImplementedError

    @abc.abstractmethod
    def global_conv(self, features, coord_features, row_splits, filters_out):
        raise NotImplementedError

    @abc.abstractmethod
    def global_deconv(
            self, global_features, coord_features, row_splits, filters_out):
        raise NotImplementedError


@gin.configurable
class ExpandingConvolver(Convolver):
    def __init__(self, activation=None, global_activation=None):
        self._activation = activation
        self._global_activation = global_activation

    def in_place_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        features = conv.flat_expanding_edge_conv(
            features,
            coord_features.flat_values,
            batched_neighbors.flat_values,
            batched_neighbors.nested_row_splits[-1],
            None if weights is None else weights.flat_values)
        if filters_out is not None:
            features = core.Dense(filters_out)(features)
        if self._activation is not None:
            features = self._activation(features)

        return features

    def resample_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        return self.in_place_conv(
            features, coord_features, batched_neighbors, filters_out, weights)

    def global_conv(
            self, features, coord_features, row_splits, filters_out):
        features = conv.flat_expanding_edge_conv(
            features,
            utils.flatten_leading_dims(coord_features, 2),
            None,
            row_splits
        )
        if filters_out is not None:
            features = core.Dense(filters_out)(features)
        if self._global_activation is not None:
            features = self._global_activation(features)
        return features

    def global_deconv(
            self, global_features, coord_features, row_splits, filters_out):
        return conv.flat_expanding_global_deconv(
            global_features,
            utils.flatten_leading_dims(coord_features, 2),
            row_splits,
            filters_out)


@gin.configurable
class ResnetConvolver(Convolver):
    # based on
    # https://github.com/keras-team/keras-applications/blob/master/'
    # 'keras_applications/resnet50.py
    def __init__(
            self, base_convolver=None, activation=tf.nn.relu, combine='add'):
        if base_convolver is None:
            base_convolver = ExpandingConvolver(activation=None)
        self._base = base_convolver
        self._activation = activation
        self._combine = combine

    def in_place_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        def base_conv(f):
            return self._base.in_place_conv(
                f, coord_features, batched_neighbors, filters_out, weights)

        x = features
        for _ in range(2):
            x = base_conv(features)
            x = core.batch_norm(x, scale=False)
            x = tf.keras.layers.Activation(self._activation)(x)
        x = base_conv(features)
        x = core.batch_norm(x)
        shortcut = features
        if self._combine == 'add':
            if features.shape[-1] != x.shape[-1]:
                shortcut = core.Dense(filters_out)(shortcut)
                shortcut = tf.keras.layers.BatchNormalization()(shortcut)
            x = tf.keras.layers.Add()([x, shortcut])
            return tf.keras.layers.Activation(self._activation)(x)
        elif self._combine == 'concat':
            x = tf.keras.layers.Activation(self._activation)(x)
            return tf.keras.layers.Lambda(tf.concat, arguments=dict(axis=-1))(
                [x, shortcut])

    def _resample_conv(self, features, conv_fn, activate_final=True):
        x = features
        scale = self._activation not in (tf.nn.relu, 'relu', None)
        for _ in range(2):
            x = conv_fn(features)
            x = core.batch_norm(x, scale=scale)
            x = tf.keras.layers.Activation(self._activation)(x)
        x = conv_fn(features)
        x = core.batch_norm(x)
        shortcut = conv_fn(features)
        shortcut = core.batch_norm(x)
        # could sometimes concat here...
        x = tf.keras.layers.Add()([x, shortcut])
        if activate_final:
            x = tf.keras.layers.Activation(self._activation)(x)
        return x

    def resample_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        def base_conv(f):
            return self._base.resample_conv(
                f, coord_features, batched_neighbors, filters_out, weights)
        return self._resample_conv(features, base_conv)

    def global_conv(
            self, features, coord_features, row_splits, filters_out):
        def base_conv(f):
            return self._base.global_conv(
                f, coord_features, row_splits, filters_out)
        return self._resample_conv(features, base_conv, activate_final=False)

    def global_deconv(self, features, coord_features, row_splits, filters_out):
        def base_conv(f):
            return self._base.global_deconv(
                f, coord_features, row_splits, filters_out)
        return self._resample_conv(features, base_conv)


def _activation(activation):
    if activation is None:
        return lambda x: x
    if isinstance(activation, six.string_types):
        return tf.keras.layers.Activation(
            tf.keras.activations.get(activation))
    return activation


@gin.configurable(blacklist=['x', 'filters_out'])
def simple_mlp(
        x, filters_out, n_hidden=1, filters_hidden=None,
        hidden_activation='relu',
        final_activation='relu'):
    if filters_hidden is None:
        filters_hidden = x.shape[-1]
    hidden_activation = _activation(hidden_activation)
    final_activation = _activation(final_activation)
    for _ in range(n_hidden):
        x = core.Dense(filters_hidden)(x)
        x = hidden_activation(x)
    x = core.Dense(filters_out)(x)
    return final_activation(x)


@gin.configurable
class NetworkConvolver(Convolver):
    def __init__(
            self, local_network_fn=simple_mlp,
            global_network_fn=simple_mlp):
        self._local_fn = local_network_fn
        self._global_fn = global_network_fn

    def in_place_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        features = conv.mlp_edge_conv(
            features,
            coord_features.flat_values,
            batched_neighbors.flat_values,
            batched_neighbors.nested_row_splits[-1],
            lambda features: self._local_fn(features, filters_out),
            None if weights is None else weights.flat_values)

        return features

    def resample_conv(
            self, features, coord_features, batched_neighbors, filters_out,
            weights):
        return self.in_place_conv(
            features, coord_features, batched_neighbors, filters_out, weights)

    def global_conv(
            self, features, coord_features, row_splits, filters_out):
        return conv.mlp_edge_conv(
            features,
            utils.flatten_leading_dims(coord_features, 2),
            None,
            row_splits,
            lambda features: self._global_fn(features, filters_out),
            weights=None)

    def global_deconv(
            self, global_features, coord_features, row_splits, filters_out):
        raise NotImplementedError('TODO')
        return conv.mlp_global_deconv(
            global_features,
            utils.flatten_leading_dims(coord_features, 2),
            row_splits,
            lambda features, global_features: self._global_network_factory(
                features, global_features, filters_out))
