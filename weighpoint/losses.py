from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from weighpoint import tf_compat  # noqa


def padded_sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=False, padding_value=-1):
    # assumes y_true is padded, y_pred is flat_values
    # y_true = tf.RaggedTensor.from_tensor(y_true, padding=padding_value)
    # y_true = y_true.flat_values
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, padding_value))

    return tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=from_logits)


class PaddedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(
            self, from_logits=True, padding_value=-1,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name=None):
        self._from_logits = from_logits
        self._padding_value = padding_value
        super(PaddedSparseCategoricalCrossentropy, self).__init__(
            reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        return padded_sparse_categorical_crossentropy(
            y_true, y_pred,
            from_logits=self._from_logits, padding_value=self._padding_value)

    def get_config(self):
        config = dict(
            from_logits=self._from_logits, padding_value=self._padding_value)
        base_config = super(
            PaddedSparseCategoricalCrossentropy, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
