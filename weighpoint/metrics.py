from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.metrics import squeeze_or_expand_dimensions
import weighpoint.tf_compat  # noqa


def padded_sparse_categorical_accuracy(y_true, y_pred, padding_value=-1):
    # assumes y_true is padded, y_pred is flat_values
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, padding_value))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


class IntersectionOverUnion(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.0, name='iou', dtype=None):
        super(IntersectionOverUnion, self).__init__(name=name, dtype=dtype)
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

        if sample_weight is not None:
            intersection = intersection * sample_weight
            union = union * sample_weight

        intersection_update = self.intersection.assign_add(
            tf.reduce_sum(intersection))
        union_update = self.union.assign_add(tf.reduce_sum(union))
        return tf.group([intersection_update, union_update])

    def result(self):
        return self.intersection / self.union


class MeanIntersectionOverUnion(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mIoU', dtype=None):
        super(MeanIntersectionOverUnion, self).__init__(
            name=name, dtype=dtype)
        self.num_classes = num_classes
        self.ious = tuple(IntersectionOverUnion(
                threshold=None, name='iou%d' % i, dtype=dtype)
            for i in range(num_classes))

    def get_config(self):
        config = super(MeanIntersectionOverUnion, self).get_config()
        assert('num_classes' not in config)
        config['num_classes'] = self.threshold
        return config

    def probs_to_preds(self, y_true, y_pred):
        assert(y_pred.dtype.is_floating)
        y_pred = tf.argmax(y_pred, axis=-1)
        return y_true, y_pred

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
            y_pred, y_true, sample_weight)
        y_true, y_pred = self.probs_to_preds(y_true, y_pred)
        updates = []
        for i, iou in enumerate(self.ious):
            updates.append(iou.update_state(
                tf.equal(y_true, i), tf.equal(y_pred, i), sample_weight))
        return tf.group(updates)

    def result(self):
        return tf.reduce_mean([iou.result() for iou in self.ious])


class PaddedMetric(tf.keras.metrics.Metric):
    def __init__(self, base_metric, padding_value=-1, name=None):
        if not isinstance(base_metric, tf.keras.metrics.Metric):
            base_metric = tf.keras.metrics.Metric.from_config(base_metric)
        if name is None:
            name = 'padded-%s' % (base_metric.name)
        super(PaddedMetric, self).__init__(name=name, dtype=base_metric.dtype)
        self._base_metric = base_metric
        self._padding_value = padding_value

    def get_config(self):
        config = super(PaddedMetric, self).get_config()
        assert(not any(k in config for k in ('base_metric', 'padding_value')))
        config['padding_value'] = self._padding_value
        config['base_metric'] = self._base_metric.get_config()

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError()
        y_true = tf.boolean_mask(
            y_true, tf.not_equal(y_true, self._padding_value))
        y_true = tf.expand_dims(y_true, axis=-1)
        return self._base_metric.update_state(y_true, y_pred)

    def result(self):
        return self._base_metric.result()


def PaddedSparseCategoricalAccuracy(padding_value=-1, dtype=None):
    return PaddedMetric(
        tf.keras.metrics.SparseCategoricalAccuracy(dtype=dtype),
        padding_value=padding_value)


def PaddedMeanIntersectionOverUnion(
        num_classes, padding_value=-1, dtype=None):
    return PaddedMetric(
        MeanIntersectionOverUnion(num_classes, dtype=dtype),
        padding_value=padding_value)
