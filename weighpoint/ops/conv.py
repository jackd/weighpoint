from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from weighpoint.ops import utils


def _reduce_flat_sum(x, row_splits_or_k, axis=0):
    assert(isinstance(x, tf.Tensor))
    if row_splits_or_k.shape.ndims == 0:
        if axis < 0:
            axis += x.shape.ndims
        k = row_splits_or_k
        shape = x.shape.as_list()
        shape.insert(axis+1, k)
        if shape[axis] is not None:
            shape[axis] = shape[axis] // k
        shape = [-1 if s is None else s for s in shape]
        x = tf.reshape(x, shape)
        return tf.reduce_sum(x, axis=axis+1)
    else:
        assert(row_splits_or_k.shape.ndims == 1)
        return utils.reduce_sum_ragged(x, row_splits_or_k)


def reduce_flat_mean(x, row_splits_or_k, weights, eps=1e-7):
    s = _reduce_flat_sum(x, row_splits_or_k)
    if weights is None:
        if row_splits_or_k.shape.ndims == 0:
            w = row_splits_or_k
        else:
            w = tf.expand_dims(utils.diff(row_splits_or_k), axis=-1)
        w = tf.cast(w, s.dtype)
    else:
        w = _reduce_flat_sum(weights, row_splits_or_k)
    if eps is not None:
        w = w + eps
    return s / w


def flat_expanding_edge_conv(
        node_features, coord_features, indices, row_splits_or_k, weights,
        eps=1e-7):
    """
    Variation of `expanding_edge_conv`.

    Args:
        node_features: [pi, fi]
        indices: [pok] or `None`. If `None`, pi == pok must be True
        coord_features: [pok, fk]

    Returns:
        convolved node_features, [pok, fi*fk]
    """
    if node_features is None:
        return reduce_flat_mean(coord_features, row_splits_or_k, weights)
    else:
        assert(all(isinstance(t, tf.Tensor) and t.shape.ndims == 2 for t in
               (node_features, coord_features)))
        assert(
            weights is None or
            isinstance(weights, tf.Tensor) and weights.shape.ndims == 2)
        if indices is not None:
            assert(indices.shape.ndims == 1)
            node_features = tf.gather(node_features, indices)
        if weights is not None:
            coord_features = weights * coord_features
        merged = utils.merge_features(node_features, coord_features)
        merged = utils.flatten_final_dims(merged, 2)
        return reduce_flat_mean(merged, row_splits_or_k, weights, eps=1e-7)


def flat_expanding_global_deconv(
        global_features, coord_features, row_splits_or_k):
    """
    Global deconvolution operation.

    Args:
        global_features: [pi, fi]
        coord_features: [po, fk]
        row_splits_or_k: [pi+1]

    Returns:
        convolved features: [po, fi*fk]
    """
    from tensorflow.python.ops.ragged.ragged_util import repeat
    if row_splits_or_k.shape.ndims == 0:
        raise NotImplementedError

    global_features = repeat(
        global_features, utils.diff(row_splits_or_k), axis=0)
    merged = utils.merge_features(global_features, coord_features)
    merged = utils.flatten_final_dims(merged, 2)
    return merged
