from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin
import numpy as np
import tensorflow as tf
from weighpoint.models import core
from weighpoint.layers import utils
from weighpoint.layers import sample
from weighpoint.layers import core as core_layers
from weighpoint.meta import builder as b
from weighpoint.models import convolvers as c
from weighpoint.models import transformers as t
from weighpoint.models import neigh as n


def _repeat(args, axis=0):
    from tensorflow.python.ops.ragged.ragged_util import repeat
    data, repeats = args
    return repeat(data, repeats, axis=axis)


def repeat(data, repeats, axis=0):
    return tf.keras.layers.Lambda(
        _repeat, arguments=dict(axis=axis))([data, repeats])


@gin.configurable
def seg_activation(x):
    return tf.keras.layers.Lambda(tf.nn.relu)(x)


def _get_class_embeddings(class_indices, num_obj_classes, filters):
    obj_class_embedder = tf.keras.layers.Embedding(
        num_obj_classes, output_dim=sum(filters), input_length=1)
    class_features = obj_class_embedder(utils.lambda_call(
        tf.expand_dims, class_indices, axis=-1))
    class_features = utils.lambda_call(tf.squeeze, class_features, axis=1)
    class_features = utils.lambda_call(
        tf.split, class_features, filters, axis=-1)
    return class_features


def _neg_inf_like(x):
    return tf.constant(-np.inf, dtype=x.dtype)*tf.ones_like(x)


@gin.configurable(
        blacklist=['inputs', 'output_spec'])
def segmentation_logits(
        inputs, output_spec, num_obj_classes=16,
        r0=0.1, initial_filters=(16,), initial_activation=seg_activation,
        filters=(32, 64, 128, 256), global_units=512,
        query_fn=core.query_pairs, radii_fn=core.constant_radii,
        global_deconv_all=True,
        coords_transform=None, weights_transform=None, convolver=None):

    if convolver is None:
        convolver = c.ExpandingConvolver(activation=seg_activation)
    if coords_transform is None:
        coords_transform = t.polynomial_transformer()
    if weights_transform is None:
        def weights_transform(*args, **kwargs):
            return None
    coords = inputs['positions']
    normals = inputs.get('normals')

    if normals is None:
        raise NotImplementedError()
    features = b.as_batched_model_input(normals)
    for f in initial_filters:
        features = tf.ragged.map_flat_values(
            core_layers.Dense(f), features)
        features = tf.ragged.map_flat_values(initial_activation, features)
    assert(isinstance(features, tf.RaggedTensor) and features.ragged_rank == 1)

    # class_embeddings = _get_class_embeddings(
    #     b.as_batched_model_input(inputs['obj_label']),
    #     num_obj_classes, [initial_filters[-1], filters[0]])

    # features = core.add_local_global(features, class_embeddings[0])

    input_row_splits = features.row_splits
    features = utils.flatten_leading_dims(features, 2)

    n_res = len(filters)
    unscaled_radii2 = radii_fn(n_res)

    if isinstance(unscaled_radii2, tf.Tensor):
        assert(unscaled_radii2.shape == (n_res,))
        radii2 = utils.lambda_call(tf.math.scalar_mul, r0**2, unscaled_radii2)
        radii2 = tf.keras.layers.Lambda(
            tf.unstack, arguments=dict(axis=0))(radii2)
        for i, radius2 in enumerate(radii2):
            tf.compat.v1.summary.scalar(
                'r%d' % i, tf.sqrt(radius2), family='radii')
    else:
        radii2 = unscaled_radii2 * (r0**2)

    def maybe_feed(r2):
        if isinstance(r2, (tf.Tensor, tf.Variable)):
            return b.prebatch_feed(tf.keras.layers.Lambda(tf.sqrt)(r2))
        else:
            return np.sqrt(r2)

    pp_radii = [maybe_feed(r2) for r2 in radii2]

    all_features = []
    in_place_neighborhoods = []
    sampled_neighborhoods = []
    global_features = []
    # encoder
    for i, (radius2, pp_radius) in enumerate(zip(radii2, pp_radii)):
        neighbors, sample_rate = query_fn(
            coords, pp_radius, name='query%d' % i)
        if not isinstance(radius2, tf.Tensor):
            radius2 = utils.constant(radius2, dtype=tf.float32)
        neighborhood = n.InPlaceNeighborhood(coords, neighbors)
        in_place_neighborhoods.append(neighborhood)
        features, nested_row_splits = core.convolve(
            features, radius2, filters[i], neighborhood, coords_transform,
            weights_transform, convolver.in_place_conv)

        all_features.append(features)

        if global_units == 'combined':
            coord_features = coords_transform(neighborhood.out_coords, None)
            global_features.append(convolver.global_conv(
                features, coord_features, nested_row_splits[-2], filters[i]))
            global_features = tf.keras.layers.Lambda(
                tf.concat, arguments=dict(axis=-1))(global_features)

        # resample
        if i < n_res - 1:
            sample_indices = sample.sample(
                sample_rate,
                tf.keras.layers.Lambda(lambda s: tf.size(s) // 4)(sample_rate))
            neighborhood = n.SampledNeighborhood(neighborhood, sample_indices)
            sampled_neighborhoods.append(neighborhood)

            features, nested_row_splits = core.convolve(
                features, radius2, filters[i+1], neighborhood,
                coords_transform, weights_transform, convolver.resample_conv)

            coords = neighborhood.out_coords

    # global_conv
    if global_units is not None:
        row_splits = nested_row_splits[-2]
        if global_units == 'combined':
            global_features = tf.keras.layers.Lambda(
                tf.concat, arguments=dict(axis=-1))(global_features)
        else:
            coord_features = coords_transform(coords, None)
            global_features = convolver.global_conv(
                features, coord_features, row_splits, global_units)

        coord_features = coords_transform(coords, None)
        features = convolver.global_deconv(
            global_features, coord_features, row_splits, filters[-1])

    # decoder
    for i in range(n_res-1, -1, -1):
        if i < n_res - 1:
            # up-sample
            neighborhood = sampled_neighborhoods.pop().transpose
            features, nested_row_splits = core.convolve(
                features, radius2, filters[i], neighborhood,
                coords_transform, weights_transform, convolver.resample_conv)
            if global_deconv_all:
                coords = neighborhood.out_coords
                row_splits = \
                    neighborhood.offset_batched_neighbors.nested_row_splits[-2]
                coord_features = coords_transform(coords, None)
                deconv_features = convolver.global_deconv(
                    global_features, coord_features, row_splits, filters[i])
                features = tf.keras.layers.Add()([features, deconv_features])

        forward_features = all_features.pop()
        if not (i == n_res-1 and global_units is None):
            features = tf.keras.layers.Lambda(
                tf.concat, arguments=dict(axis=-1))(
                    [features, forward_features])
        neighborhood = in_place_neighborhoods.pop().transpose
        features, nested_row_splits = core.convolve(
            features, radius2, filters[i], neighborhood,
            coords_transform, weights_transform, convolver.resample_conv)

    logits = core_layers.Dense(output_spec.shape[-1])(features)

    valid_classes_mask = inputs.get('valid_classes_mask')
    if valid_classes_mask is not None:
        # overwrite invalid class logits with -infinity
        row_lengths = utils.diff(input_row_splits)
        valid_classes_mask = b.as_batched_model_input(valid_classes_mask)
        valid_classes_mask = repeat(valid_classes_mask, row_lengths, axis=0)

        neg_inf = tf.keras.layers.Lambda(_neg_inf_like)(logits)
        logits = utils.lambda_call(
            tf.where, valid_classes_mask, logits, neg_inf)

    return logits
