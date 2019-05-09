from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


def continuous_iou_loss(y_true, y_pred, from_logits=True):
    with tf.name_scope('continuous_iou'):
        if from_logits:
            y_pred = tf.nn.softmax(y_pred)
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int64)
        indices = tf.range(tf.size(y_true, out_type=tf.int64), dtype=tf.int64)
        indices = tf.stack((indices, y_true), axis=1)
        intersection = tf.gather_nd(y_pred, indices)
        union = tf.reduce_sum(y_pred, axis=-1) - intersection + 1.0
        iou = intersection / union
        loss = -iou
    return loss


@gin.configurable
class ContinousIouLoss(tf.keras.losses.Loss):
    def __init__(
            self, from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name=None):
        self._from_logits = from_logits
        super(ContinousIouLoss, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super(
            ContinousIouLoss, self).get_config()
        config['from_logits'] = self._from_logits
        return config

    def call(self, y_true, y_pred):
        return continuous_iou_loss(
            y_true, y_pred, from_logits=self._from_logits)
