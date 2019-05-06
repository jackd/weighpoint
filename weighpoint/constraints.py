from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin


@gin.configurable(module='constraints')
class CompoundConstraint(tf.keras.constraints.Constraint):
    def __init__(self, constraints):
        self._constraints = tuple(constraints)

    def __call__(self, w):
        for c in self._constraints:
            w = c(w)
        return w

    @property
    def constraints(self):
        return self._constraints

    def get_config(self):
        return dict(constraints=[c.get_config() for c in self._constraints])


def compound_constraint(*constraints):
    if len(constraints) == 0:
        return None
    elif len(constraints) == 1:
        return constraints[0]
    else:
        return CompoundConstraint(constraints)


@gin.configurable(module='constraints')
class MaxValue(tf.keras.constraints.Constraint):
    def __init__(self, value):
        self._value = value

    def get_config(self):
        return dict(value=self._value)

    @property
    def value(self):
        return self._value

    def __call__(self, w):
        return tf.maximum(w, self._value)


@gin.configurable(module='constraints')
class WeightDecay(tf.keras.constraints.Constraint):
    """Equivalent to regularizers.l2(decay/2) when using SGD."""
    def __init__(self, decay=0.02):
        self._factor = 1 - decay
        self._decay = decay

    def get_config(self):
        return dict(decay=self._decay)

    def __call__(self, w):
        if self._factor != 1:
            w = self._factor * w
        return w
