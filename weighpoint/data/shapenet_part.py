from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gin
from tensorflow_datasets.volume.shapenet import iccv2017
from weighpoint.data import problems
# from weighpoint.losses import PaddedSparseCategoricalCrossentropy
# from weighpoint.metrics import PaddedSparseCategoricalAccuracy
from weighpoint.metrics import MeanIntersectionOverUnion
from weighpoint.metrics import IndividualIntersectionOverUnion
from weighpoint.meta import builder as b


CLASS_FREQ = (
    2275419,
    1664032,
    642156,
    460568,
    9715,
    138138,
    75487,
    26726,
    97844,
    131510,
    288766,
    1302657,
    2430900,
    2989559,
    1510814,
    260298,
    74972,
    31482,
    15683,
    109676,
    246091,
    942116,
    298731,
    289889,
    357113,
    1535899,
    30956,
    506672,
    479080,
    415252,
    15960,
    13613,
    80202,
    5217,
    2217,
    223851,
    22325,
    343634,
    359971,
    164593,
    30528,
    76608,
    19519,
    12561,
    29492,
    218704,
    18829,
    7729973,
    2395978,
    317260,
)


def get_part_masks():
    out = np.zeros(shape=(
        iccv2017.NUM_OBJECT_CLASSES, iccv2017.NUM_PART_CLASSES), dtype=np.bool)
    for i in range(iccv2017.NUM_OBJECT_CLASSES):
        out[i, iccv2017.LABEL_SPLITS[i]:iccv2017.LABEL_SPLITS[i+1]] = True
    return tf.constant(out, dtype=tf.bool)


@gin.configurable
class UniformShapenetPartConfig(iccv2017.ShapenetPart2017Config):
    def __init__(self, r0=0.1, k=10, class_id=None):
        self._r0 = r0
        self._k = k
        name_prefix = 'rescaled-%d-%.2f' % (k, r0)
        super(UniformShapenetPartConfig, self).__init__(
            name_prefix=name_prefix,
            description='Point cloud segmentation dataset with mean %dth '
                        'neighbor at distance %.2f' % (k, r0),
            version='0.0.1',
            class_id=class_id)

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
            self, map_fn=None, shuffle_buffer=1024, download_and_prepare=True,
            loss=problems.SparseCategoricalCrossentropy(from_logits=True),
            use_inverse_freq_weights=False, individual_iou_metrics=False,
            class_id=None):
        builder = get_part_builder()
        self._builder = builder
        self._use_inverse_freq_weights = use_inverse_freq_weights
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
            MeanIntersectionOverUnion(
                num_classes=self._num_classes(), ignore_weights=True)]
        if individual_iou_metrics:
            metrics += [
                IndividualIntersectionOverUnion(
                    i, name='iou%d' % i, ignore_weights=True) for i in
                range(builder.info.features['cloud']['labels'].num_classes)]

        super(ShapenetPartProblem, self).__init__(
            builder=builder,
            # loss=PaddedSparseCategoricalCrossentropy(from_logits=True),
            loss=loss,
            metrics=metrics,
            map_fn=map_fn, as_supervised=True,
            shuffle_buffer=shuffle_buffer,
            download_and_prepare=download_and_prepare)

    def preprocess_labels(self, labels, weights=None):
        labels = b.batched(labels).flat_values
        if weights is None:
            return labels, None
        else:
            return labels, b.batched(weights).flat_values

    def _num_classes(self):
        lower, upper = self._class_range()
        return upper - lower

    def _class_range(self):
        ci = self._builder.builder_config.class_index
        if ci is None:
            return 0, iccv2017.LABEL_SPLITS[-1]
        return iccv2017.LABEL_SPLITS[ci], iccv2017.LABEL_SPLITS[ci+1]

    def output_spec(self):
        return tf.keras.layers.InputSpec(
            dtype=tf.float32, shape=(self._num_classes(),))

    def data_pipeline(self, dataset, split, batch_size, prefetch=True):

        def initial_map_fn(inputs, labels):
            point_labels = inputs.pop('labels')
            mapped_inputs = dict(
                positions=inputs['positions'],
                normals=inputs['normals'],
                obj_label=labels)
            if self._builder.builder_config.class_index is None:
                masks = get_part_masks()
                mask = tf.gather(masks, labels)
                mapped_inputs['valid_classes_mask'] = mask
            for k in inputs:
                if k not in mapped_inputs:
                    mapped_inputs[k] = inputs[k]

            return mapped_inputs, point_labels

        map_fn = self._map_fn
        if isinstance(map_fn, dict):
            map_fn = map_fn[split]
        # if split == tfds.Split.TRAIN:
        dataset = dataset.shuffle(self._shuffle_buffer)
        lower, upper = self._class_range()
        class_freq = np.array(CLASS_FREQ[lower: upper], dtype=np.float32)

        def actual_map_fn(inputs, labels):
            inputs, labels = initial_map_fn(inputs, labels)

            if map_fn is not None:
                mapped_inputs, labels = map_fn(inputs, labels)
                for k in inputs:
                    if k not in mapped_inputs:
                        mapped_inputs[k] = inputs[k]
                inputs = mapped_inputs

            if self._use_inverse_freq_weights:
                weights = 1. / np.array(class_freq)
                weights *= np.mean(class_freq)
                weights = tf.constant(weights, dtype=tf.float32)
                weights = tf.gather(weights, labels)
                return inputs, labels, weights
            else:
                return inputs, labels

        dataset = dataset.map(actual_map_fn, tf.data.experimental.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        if prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
