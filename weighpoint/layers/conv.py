from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from weighpoint.ops import conv as conv_ops
from weighpoint.layers import utils


def flat_expanding_edge_conv(
        node_features, coord_features, indices, row_splits_or_k, weights=None):
    features = utils.lambda_call(
        conv_ops.flat_expanding_edge_conv, node_features, coord_features,
        indices, row_splits_or_k, weights)
    return features


def flat_expanding_global_deconv(
        global_features, coord_features, row_splits_or_k):
    features = utils.lambda_call(
        conv_ops.flat_expanding_global_deconv, global_features, coord_features,
        row_splits_or_k)
    return features


def reduce_flat_mean(x, row_splits_or_k, weights, eps=1e-7):
    return utils.lambda_call(
        conv_ops.reduce_flat_mean, x, row_splits_or_k, weights, eps=eps)


def mlp_edge_conv(
        node_features, coord_features, indices, row_splits_or_k,
        network_fn, weights=None):
    if indices is not None:
        node_features = utils.gather(node_features, indices)
    features = tf.keras.layers.Lambda(tf.concat, arguments=dict(axis=-1))(
        [node_features, coord_features])
    features = network_fn(features)
    return reduce_flat_mean(features, row_splits_or_k, weights)


def _expand_and_tile(global_features, row_splits_or_k):
    if not isinstance(row_splits_or_k, tf.Tensor) or (
            row_splits_or_k.shape.ndims == 0):
        # knn
        raise NotImplementedError('TODO')
    else:
        from tensorflow.python.ops.ragged.ragged_util import repeat
        return repeat(global_features, row_splits_or_k, axis=0)


def mlp_global_deconv(
        global_features, coord_features, row_splits_or_k, network_fn):
    global_features = utils.lambda_call(
        _expand_and_tile, global_features, row_splits_or_k)
    features = tf.keras.layers.Lambda(tf.concat, arguments=dict(axis=-1))([
        global_features, coord_features])
    return network_fn(features)
