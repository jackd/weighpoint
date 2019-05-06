from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin
from tensorflow_datasets.volume.shapenet import iccv2017
from weighpoint.data import problems
from weighpoint.losses import PaddedSparseCategoricalCrossentropy
from weighpoint.metrics import PaddedSparseCategoricalAccuracy
from weighpoint.metrics import PaddedMeanIntersectionOverUnion


def get_part_masks():
    out = np.zeros(shape=(
        iccv2017.NUM_OBJECT_CLASSES, iccv2017.NUM_PART_CLASSES), dtype=np.bool)
    for i in range(iccv2017.NUM_OBJECT_CLASSES):
        out[i, iccv2017.LABEL_SPLITS[i]:iccv2017.LABEL_SPLITS[i+1]] = True
    return tf.constant(out, dtype=tf.bool)


@gin.configurable
class UniformShapenetPartConfig(iccv2017.ShapenetPart2017Config):
    def __init__(self, r0=0.1, k=10):
        self._r0 = r0
        self._k = k
        name = 'rescaled-%d-%.2f' % (k, r0)
        super(UniformShapenetPartConfig, self).__init__(
            name=name,
            description='Point cloud segmentation dataset with mean %dth '
                        'neighbor at distance %.2f' % (k, r0),
            version='0.0.1')

    def map_cloud(self, cloud):
        from scipy.spatial import cKDTree
        positions = cloud['positions']
        tree = cKDTree(positions)
        dists, indices = tree.query(tree.data, self._k)
        new_scale_factor = self._r0 / np.mean(dists[:, -1])
        positions *= new_scale_factor
        cloud['positions'] = positions
        return cloud


@gin.configurable
def get_part_builder(config=None):
    if config is None:
        config = UniformShapenetPartConfig()
    builder = iccv2017.ShapenetPart2017(config=config)
    return builder


@gin.configurable
class ShapenetPartProblem(problems.TfdsProblem):
    def __init__(
            self, map_fn=None, shuffle_buffer=1024, download_and_prepare=True):
        builder = get_part_builder()

        super(ShapenetPartProblem, self).__init__(
            builder=builder,
            loss=PaddedSparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                PaddedSparseCategoricalAccuracy(),
                PaddedMeanIntersectionOverUnion(
                    num_classes=iccv2017.NUM_PART_CLASSES)],
            map_fn=map_fn, as_supervised=True,
            shuffle_buffer=shuffle_buffer,
            download_and_prepare=download_and_prepare)

    def output_spec(self):
        return tf.keras.layers.InputSpec(
            dtype=tf.float32, shape=(iccv2017.NUM_PART_CLASSES,))

    def data_pipeline(self, dataset, split, batch_size, prefetch=True):
        masks = get_part_masks()

        def initial_map_fn(inputs, labels):
            point_labels = inputs.pop('labels')
            mask = tf.gather(masks, labels)
            mapped_inputs = dict(
                positions=inputs['positions'],
                normals=inputs['normals'],
                obj_label=labels,
                valid_classes_mask=mask)
            for k in inputs:
                if k not in mapped_inputs:
                    mapped_inputs[k] = inputs[k]
            return mapped_inputs, point_labels

        map_fn = self._map_fn
        if isinstance(map_fn, dict):
            map_fn = map_fn[split]
        # if split == tfds.Split.TRAIN:
        dataset = dataset.repeat().shuffle(self._shuffle_buffer)

        def actual_map_fn(inputs, labels):
            inputs, labels = initial_map_fn(inputs, labels)
            if map_fn is not None:
                mapped_inputs, labels = map_fn(inputs, labels)
                for k in inputs:
                    if k not in mapped_inputs:
                        mapped_inputs[k] = inputs[k]
                return mapped_inputs, labels
            else:
                return inputs, labels

        dataset = dataset.map(actual_map_fn, tf.data.experimental.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        if prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
