from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import six
import tensorflow as tf


def is_namedtuple(x):
    return (isinstance(x, tuple) and
            isinstance(getattr(x, '__dict__', None), collections.Mapping) and
            getattr(x, '_fields', None) is not None)


def yield_flat_paths(nest):
    """tf.nest.yield_flat_paths isn't in tf 1.14 or 2.0."""
    if isinstance(nest, (dict, collections.Mapping)):
        for key in sorted(nest):
            value = nest[key]
            for sub_path in yield_flat_paths(value):
                yield (key,) + sub_path
    elif is_namedtuple(nest):
        for key in nest._fields:
            value = getattr(nest, key)
            for sub_path in yield_flat_paths(value):
                yield (key,) + sub_path
    elif isinstance(nest, six.string_types):
        yield ()
    elif isinstance(nest, collections.Sequence):
        for idx, value in enumerate(nest):
            for sub_path in yield_flat_paths(value):
                yield (idx,) + sub_path
    else:
        yield ()


def assert_all_tensors(args):
    for i, a in enumerate(args):
        if not isinstance(a, tf.Tensor):
            raise ValueError('expected all tensors, but arg %d is %s' % (i, a))


class Finalizable(object):
    @abc.abstractproperty
    def finalize(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def finalized(self):
        raise NotImplementedError()

    def _assert_mutable(self, action):
        if self.finalized:
            raise RuntimeError('ModelBuilder finalized - cannot %s' % action)

    def _assert_finalized(self, action):
        if not self.finalized:
            raise RuntimeError(
                'ModelBuilder not finalized - cannot %s' % action)
