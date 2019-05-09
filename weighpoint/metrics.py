from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.metrics import squeeze_or_expand_dimensions
import weighpoint.tf_compat  # noqa


class IntersectionOverUnion(tf.keras.metrics.Metric):
    def __init__(
            self, threshold=0.0, ignore_weights=False, name='iou', dtype=None):
        super(IntersectionOverUnion, self).__init__(name=name, dtype=dtype)
        self.ignore_weights = ignore_weights
        self.threshold = threshold
        self.intersection = self.add_weight(
            'intersection',
            shape=(),
            initializer=tf.keras.initializers.Zeros())
        self.union = self.add_weight(
            'union',
            shape=(),
            initializer=tf.keras.initializers.Zeros())

    def get_config(self):
        config = super(IntersectionOverUnion, self).get_config()
        assert('threshold' not in config)
        config['threshold'] = self.threshold
        config['ignore_weights'] = self.ignore_weights
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
            y_pred, y_true, sample_weight)
        if self.threshold is None:
            assert(y_pred.dtype.is_bool)
        else:
            assert(y_pred.dtype.is_floating)
            y_pred = tf.greater(y_pred, self.threshold)
        y_true = tf.cast(y_true, tf.bool)
        intersection = tf.cast(tf.logical_and(y_true, y_pred), self._dtype)
        union = tf.cast(tf.logical_or(y_true, y_pred), self._dtype)

        if sample_weight is not None and not self.ignore_weights:
            intersection = intersection * sample_weight
            union = union * sample_weight

        intersection_update = self.intersection.assign_add(
            tf.reduce_sum(intersection))
        union_update = self.union.assign_add(tf.reduce_sum(union))
        return tf.group([intersection_update, union_update])

    def result(self):
        return self.intersection / self.union


class IndividualIntersectionOverUnion(IntersectionOverUnion):
    def __init__(self, index, **kwargs):
        self._index = index
        super(IndividualIntersectionOverUnion, self).__init__(
            threshold=None, **kwargs)

    def get_config(self):
        config = super(IndividualIntersectionOverUnion, self).get_config()
        config['index'] = self._index
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError()
        y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
            y_pred, y_true, sample_weight)
        y_true = tf.cast(y_true, tf.int64)
        y_pred = tf.equal(tf.argmax(y_pred, axis=-1), self._index)
        y_true = tf.equal(y_true, self._index)
        return super(IndividualIntersectionOverUnion, self).update_state(
            y_true, y_pred, sample_weight)


class MeanIntersectionOverUnion(tf.keras.metrics.Metric):
    def __init__(
            self, num_classes, ignore_weights=False, name='mIoU', dtype=None):
        super(MeanIntersectionOverUnion, self).__init__(
            name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ignore_weights = ignore_weights
        self.ious = tuple(IntersectionOverUnion(
                threshold=None, name='iou%d' % i, dtype=dtype,
                ignore_weights=ignore_weights)
            for i in range(num_classes))

    def get_config(self):
        config = super(MeanIntersectionOverUnion, self).get_config()
        assert(all(k not in config for k in ('num_classes', 'ignore_weights')))
        config['num_classes'] = self.num_classes
        config['ignore_weights'] = self.ignore_weights
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
            y_pred, y_true, sample_weight)
        y_pred = tf.argmax(y_pred, axis=-1)
        updates = []
        for i, iou in enumerate(self.ious):
            updates.append(iou.update_state(
                tf.equal(y_true, i), tf.equal(y_pred, i), sample_weight))
        return tf.group(tf.nest.flatten(updates))

    def result(self):
        return tf.reduce_mean(tf.stack([iou.result() for iou in self.ious]))
