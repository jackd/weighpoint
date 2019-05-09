from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin
from weighpoint.losses.lovasz import original as _orig


def lovasz_loss(y_true, y_pred, classes='present', from_logits=False):
    if from_logits:
        y_pred = tf.nn.softmax(y_pred)
    if y_true.shape.ndims == y_pred.shape.ndims:
        y_true = tf.squeeze(y_true, axis=-1)
    ndims = y_pred.shape.ndims
    if ndims == 4:
        return _orig.lovasz_softmax(
            y_pred, y_true, classes=classes, per_image=False)
    else:
        assert(ndims == 2)
        return _orig.lovasz_softmax_flat(y_pred, y_true, classes=classes)


@gin.configurable
class LovaszLoss(tf.keras.losses.Loss):
    def __init__(
            self, from_logits=True, classes='present',
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name=None):
        self._from_logits = from_logits
        self._classes = classes
        super(LovaszLoss, self).__init__(reduction=reduction, name=name)

    def get_config(self):
        config = super(LovaszLoss, self).get_config()
        config['from_logits'] = self._from_logits
        config['classes'] = self._classes
        return config

    def call(self, y_true, y_pred):
        return lovasz_loss(
            y_true, y_pred, from_logits=self._from_logits,
            classes=self._classes)
