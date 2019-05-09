from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf
from weighpoint.meta import model_builder as mb
from weighpoint.meta import utils as meta_utils


class RaggedBatcher(object):
    def __init__(self, prebatch, postbatch):
        self._prebatch = prebatch
        self._postbatch = postbatch


class MetaNetworkBuilder(meta_utils.finalizable):
    def __init__(self):
        self._prebatch = mb.ModelBuilder('prebatch')
        self._postbatch = mb.ModelBuilder('postbatch')
        self._learned = mb.ModelBuilder('learned')
        self._batcher = RaggedBatcher(self._prebatch, self._postbatch)
        self._labels = []
        self._finalized = False

    def __enter__(self):
        _builders.append(self)

    def __exit__(self, *args, **kwargs):
        last = _builders.pop()
        assert(last is self)

    def finalize(self):
        if self._finalized:
            return
        for builder in (self._prebatch, self._postbatch, self._learned):
            builder.finalize()
        self._finalized = True

    @property
    def finalized(self):
        return self._finalized

    def dataset_inputs(self, dataset):
        return mb.dataset_map_inputs(self._prebatch, dataset)


def get_default_builder():
    if len(_builders) == 0:
        raise RuntimeError(
            'No `MetaNetworBuilder` contexts open - use `with meta_builder:`')
    return _builders[-1]


_builders = []
