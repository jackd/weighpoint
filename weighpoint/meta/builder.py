from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import tensorflow as tf
from weighpoint.meta import preprocessor as p
from weighpoint.layers import utils
from weighpoint.layers import ragged
from weighpoint.tf_compat import dim_value


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


def _feed(tensor, registry):
    if not isinstance(tensor, (tf.Tensor, tf.Variable)):
        raise ValueError(
            'tensor must be a tensor or variable, got %s' % str(tensor))
    inp = tf.keras.layers.Input(
        shape=tensor.shape, dtype=tensor.dtype, batch_size=1)
    registry.append((inp, tensor))
    return tf.squeeze(inp, axis=0)


class Marks(object):
    PREBATCH = 0
    BATCHED = 1
    MODEL = 2

    _strings = ('prebatch', 'batched', 'model')

    @classmethod
    def to_string(cls, mark):
        return cls._strings[mark]


class MetaNetworkBuilder(object):
    def __init__(self):
        self._prebatch_inputs = []
        self._prebatch_outputs = []
        self._prebatch_feeds = []
        self._batched_inputs = []
        self._batched_outputs = []
        self._model_inputs = []

        self._prebatch_feed_dict = {}
        self._batched_inputs_dict = {}
        self._model_inputs_dict = {}

        self._marks = {}

    def get_mark(self, tensor):
        return self._marks.get(tensor)

    def __enter__(self):
        _builder_stack.append(self)

    def __exit__(self, *args, **kwargs):
        out = _builder_stack.pop()
        if out is not self:
            raise RuntimeError(
                'self not on top of stack when attempting to exit')

    def preprocessor(self):
        prebatch_feed_values = tuple(self._prebatch_feed_dict.keys())
        prebatch_feed_inputs = tuple(
            self._prebatch_feed_dict[k] for k in prebatch_feed_values)
        return p.Preprocessor.from_io(
            tuple(self._prebatch_inputs),
            tuple(self._prebatch_outputs),
            prebatch_feed_inputs, prebatch_feed_values,
            tuple(self._batched_inputs),
            tuple(self._batched_outputs),
        )

    def model(self, outputs):
        assert_all_tensors(outputs)
        return tf.keras.models.Model(
            inputs=tuple(self._model_inputs),
            outputs=tuple(outputs))

    def _assert_not_marked(self, tensor, mark_set, target_set):
        if tensor in self._marks[mark_set]:
            raise ValueError(
                'tensor %s is already marked as %s - cannot be marked as %s '
                'as well' % (str(tensor, mark_set, target_set)))

    def _mark(self, tensor, mark, recursive=True):
        existing = self._marks.get(tensor)
        if existing is None:
            self._marks[tensor] = mark
            if recursive:
                for dep in tensor.op.inputs:
                    self._mark(dep, mark)
        elif existing != mark:
            raise ValueError(
                'attempted to mark tensor %s as %s, but it is already marked '
                'as %s' % (
                    tensor, Marks.to_string(mark), Marks.to_string(existing)))

    def prebatch_input(self, shape, dtype, name=None):
        inp = tf.keras.layers.Input(
            shape=shape, dtype=dtype, batch_size=1, name=name)
        self._prebatch_inputs.append(inp)
        inp = utils.lambda_call(tf.squeeze, inp, axis=0)
        self._mark(inp, Marks.PREBATCH)
        return inp

    def prebatch_feed(self, tensor):
        if tensor in self._prebatch_feed_dict:
            return self._prebatch_feed_dict[tensor]
        self._mark(tensor, Marks.MODEL)
        inp = tf.keras.layers.Input(shape=tensor.shape, dtype=tensor.dtype)
        # inp = tf.keras.layers.Input(
        #     tensor=utils.lambda_call(tf.expand_dims, tensor, axis=0))
        out = utils.lambda_call(tf.squeeze, inp, axis=0)
        self._mark(out, Marks.PREBATCH, recursive=False)
        self._prebatch_feed_dict[tensor] = inp
        return out

    def prebatch_inputs_from(self, dataset):
        types = dataset.output_types[0]
        # ensure inputs are added in the correct order
        flat_shapes = tf.nest.flatten(dataset.output_shapes[0])
        flat_types = tf.nest.flatten(types)
        names = ['-'.join([str(pi) for pi in p]) for p in
                 yield_flat_paths(dataset.output_shapes[0])]
        inputs = tuple(
            self.prebatch_input(s, t, name=n) for s, t, n in zip(
                flat_shapes, flat_types, names))
        return tf.nest.pack_sequence_as(types, inputs)

    def _batched_fixed_tensor(self, tensor):
        assert(isinstance(tensor, tf.Tensor))
        if tensor in self._batched_inputs_dict:
            return self._batched_inputs_dict[tensor]
        self._mark(tensor, Marks.PREBATCH)
        self._prebatch_outputs.append(tensor)
        batched = tf.keras.layers.Input(shape=tensor.shape, dtype=tensor.dtype)
        self._batched_inputs.append(batched)
        self._mark(batched, Marks.BATCHED)
        self._batched_inputs_dict[tensor] = batched
        return batched

    def _batched_tensor(self, tensor):
        shape = tensor.shape
        if len(shape) > 0 and dim_value(shape[0]) is None:
            return self._batched_tensor_with_ragged_leading_dim(tensor)

        return self._batched_fixed_tensor(tensor)

    def _batched_tensor_with_ragged_leading_dim(self, tensor):
        assert(dim_value(tensor.shape[0]) is None)
        if tensor in self._batched_inputs_dict:
            return self._batched_inputs_dict[tensor]
        size = tf.keras.layers.Lambda(
                lambda x: tf.shape(x, out_type=tf.int64)[0])(tensor)
        values = self._batched_fixed_tensor(tensor)
        lengths = self._batched_fixed_tensor(size)
        out = ragged.ragged_from_tensor(values, lengths)
        self._batched_inputs_dict[tensor] = out
        return out

    def _batched_ragged(self, rt):
        if rt in self._batched_inputs_dict:
            return self._batched_inputs_dict[rt]
        size = self._batched_fixed_tensor(
                ragged.ragged_lambda(lambda x: x.nrows())(rt))
        nested_row_lengths = ragged.nested_row_lengths(rt)
        nested_row_lengths = [
            self._batched_tensor(rl) for rl in nested_row_lengths]
        nested_row_lengths = [
            utils.flatten_leading_dims(rl, 2) for rl in nested_row_lengths]
        values = utils.flatten_leading_dims(
            self._batched_tensor(rt.flat_values), 2)
        out = ragged.ragged_from_nested_row_lengths(
            values, [size] + nested_row_lengths)
        self._batched_inputs_dict[rt] = out
        return out

    def _batched(self, rt):
        # handle possible raggedness of possibly ragged tensor
        if isinstance(rt, tf.RaggedTensor):
            return self._batched_ragged(rt)

        assert(isinstance(rt, tf.Tensor))
        return self._batched_tensor(rt)

    def batched(self, tensor):
        return tf.nest.map_structure(self._batched, tensor)

    def _as_model_input(self, tensor):
        assert(isinstance(tensor, tf.Tensor))
        if tensor in self._model_inputs_dict:
            return self._model_inputs_dict[tensor]
        self._mark(tensor, Marks.BATCHED)
        self._batched_outputs.append(tensor)
        inp = tf.keras.layers.Input(shape=tensor.shape[1:], dtype=tensor.dtype)
        self._model_inputs.append(inp)
        self._mark(inp, Marks.MODEL)
        self._model_inputs_dict[tensor] = inp
        return inp

    def as_model_input(self, tensor):
        if isinstance(tensor, tf.Tensor):
            return self._as_model_input(tensor)
        elif isinstance(tensor, tf.RaggedTensor):
            values = self.as_model_input(tensor.values)
            row_splits = self._as_model_input(tensor.row_splits)
            return tf.RaggedTensor.from_row_splits(values, row_splits)
        else:
            raise ValueError('Unrecognized type "%s"' % tensor)

    def as_batched_model_input(self, tensor):
        return self.as_model_input(self.batched(tensor))


_base_builder = MetaNetworkBuilder()
_builder_stack = [_base_builder]


def prebatch_input(shape, dtype):
    return _builder_stack[-1].prebatch_input(shape=shape, dtype=dtype)


def prebatch_feed(tensor):
    return _builder_stack[-1].prebatch_feed(tensor)


def prebatch_inputs_from(dataset):
    return _builder_stack[-1].prebatch_inputs_from(dataset)


def batched(tensor):
    return _builder_stack[-1].batched(tensor)


def as_model_input(tensor):
    return _builder_stack[-1].as_model_input(tensor)


def as_batched_model_input(tensor):
    return _builder_stack[-1].as_batched_model_input(tensor)


def preprocessor():
    return _builder_stack[-1].preprocessor()


def model(outputs):
    return _builder_stack[-1].model(outputs)


def get_mark(tensor):
    return _builder_stack[-1].get_mark(tensor)
