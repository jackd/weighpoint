"""
Provides tools for building meta-networks.

Meta-network building allows code associated with conceptual layers to be
written in one place, even when it involves pre-batch, post-batch and main
network operations. This is useful when you have networks with per-layer
preprocessing.

For example, consider the following code.

```python

class MyLayerBuilder(object):
    def prebatch_map(self, example_data):
        ...

    def network_fn(self, batch_data, batched_preprocessed_data):
        ...


layer1_builder = MyLayerBuilder()
layer1_builder = MyLayerBuilder()


def prebatch_preprocess(original_inputs):
    x = original_inputs
    x, layer1_prep_inputs = layer_builder1.prebatch_map(x)
    _, layer1_prep_inputs = layer_builder2.prebatch_map(x)
    return original_inputs, layer1_prep_inputs, layer2_prep_inputs


def get_model(inputs):
    x, layer1_prep_inputs, layer2_prep_inputs = inputs
    x = layer1_builder.network_fn(x, layer1_prep_inputs)
    x = layer2_builder.network_fn(x, layer2_prep_inputs)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


dataset = get_original_dataset(...)
dataset = dataset.repeat().shuffle(buffer_size).map(
    lambda inputs, labels: prebatch_preprocess(inputs), labels).batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

inputs = tf.nest.map_structure(
    lambda s, d: tf.keras.layers.Input(shape=s, dtype=d),
    dataset.output_shapes, dataset.output_types)

model = get_model(inputs)
model.compile(...)
model.fit(dataset, ...)
```

This can be done using this module as follows.
```python
from weighpoint.meta import builder as b
dataset = get_original_dataset(...)

inputs = b.prebatch_inputs_from(dataset)
x = inputs
x, layer1_prep_inputs = layer1_builder.prebatch_map(x)
_, layer2_prep_inputs = layer2_builder.prebatch_map(x)

batched_layer1_prep_inputs = b.as_model_inputs(layer1_prep_inputs)
batched_layer2_prep_inputs = b.as_model_inputs(layer2_prep_inputs)
batched_inputs = b.as_model_input(inputs)

batched_x = batched_inputs
batched_x = layer1_builder.network_fn(batched_x, batched_layer1_prep_inputs)
batched_x = layer2_builder.network_fn(batched_x, batched_layer2_prep_inputs)

model = b.model(batched_x)
preprocessor = b.preprocessor()

dataset = preprocessor.map_and_batch(
    dataset.repeat().shuffle(buffer_size)).prefetch(
        tf.data.experimental.AUTOTUNE)

model.compile(...
model.fit(dataset, ...)
```
While the benefit may not be clear in this example, keep in mind this is a
simple sequential 2-layer model. The cost of developing, maintaining and
coordinating separate preprocessing and network functions grows rapidly with
model complexity. See `weighpoint.models` for example usage.

Post-batch preprocessing is also supported by using separate `b.batched` and
`b.as_model_input` calls (`b.as_batched_model_input` is a simple wrapper around
both of these).

Under the hood, this is done by keeping track of different sets of inputs and
outputs which go on to form prebatch, postbatch and learned keras models. For
example, `batched_x = b.batched(x)` marks `x` as an output of the prebatch map
model and `batched_x` as an input to the postbatch map model, while
`model_z = b.as_model_input(batched_y)` will mark `model_x` as a learned model
input and `batched_y` as an output of the postbatch map model.

In addition to this, tensors with dynamic leading dimension are automatically
converted to `tf.RaggedTensor`s. Under-the-hood, this ragged batching is done
in a 3-stage process:
1. during `batched`, the leading dimension is checked and the size is stored
    if it is dynamic
2. during `Preprocessor.batch`, `tf.data.Dataset.padded_batch` is used if
    any dynamic leading dimensions were detected.
3. during `Preprocessor.postbatch_map`, the padded values are stripped
    efficiently using the size stored in step 1 and the ragged tensor is
    constructed.

If you wish to build multiple meta-networks separately, you can create
`MetaNetworkBuilder`s and use context blocks.

```python
from weighpoint.meta import builder as b
first_builder = b.MetaNetworkBuilder()

with first_builder:
    ...
    second_builder = b.MetaNetworkBuilder()
    with second_builder:
        # b.foo() redirects to second_builder.foo()
        b1_input = b.prebatch_input(...)
        b0_input = first_builder.prebatch_input(...)
        ...

m0 = b0.model(m0_outputs)
m1 = b1.model(m1_outputs)
p0 = b0.preprocessor()
p1 = b1.preprocessor()
```
"""

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


class Marks(object):
    """
    Different marks for tensors in a meta-network.

    Each tensor in a meta-network should be part of only 1 of prebatch,
    postbatch or learned (model) networks. If, for example, you wish for a
    batched network tensor to be used as part of the model, create a new tensor
    using `b.as_model_input(batched_tensor)`.
    """
    PREBATCH = 0
    BATCHED = 1
    MODEL = 2

    _strings = ('prebatch', 'batched', 'model')

    @classmethod
    def to_string(cls, mark):
        return cls._strings[mark]


class MetaNetworkBuilder(object):
    """See `help(weighpoint.meta.builder)`."""

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
        """See `weighpoint.meta.builder.Marks`."""
        return self._marks.get(tensor)

    def __enter__(self):
        _builder_stack.append(self)

    def __exit__(self, *args, **kwargs):
        out = _builder_stack.pop()
        if out is not self:
            raise RuntimeError(
                'self not on top of stack when attempting to exit')

    def preprocessor(self):
        """Get a `weighpoint.meta.preprocessor.Preprocessor`."""
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
        """Get the learned model with the given outputs."""
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
        """
        Denote a learnable tensor as being used in the prebatch mapping.

        This allows learned model parameters (or tensors derived from them) to
        be used in the preprocessing. The resulting mapped datasets need to be
        iterated over by an reinitializable iterator and the values used will
        only be updated after each reinitialization.

        Gradients will not be propagated through the batching/mapping process.

        For example, weighpoint convolutions have a weighting function with
        associated root which doubles as the ball-search radius. We can learn
        the weighting function in the network as normal and use the resulting
        root to adapt our ball search radius.
        """
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
        """
        Get all inputs associated with the first element of dataset.

        Args:
            dataset: with (features, labels), where features and labels
                are tensor structures (lists, tuples, dicts etc.).

        Returns:
            `tf.keras.layers.Input` associated with each tensor of`features`.
        """
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
        """
        Create a structure representing the batched form of the input.

        Example usage:
        ```python
        input_a = b.prebatch_input(shape=(), dtype=tf.float32)
        input_b = b.prebatch_input(shape=(3,), dtype=tf.float32)
        x = 2*a + b  # or the `tf.keras.layers.Lambda` wrapped version in 1.x
        y = x + 1
        batched_x, batched_y = b.batched((x, y))
        print(x.shape)          # (3,)
        print(y.shape)          # (3,)
        print(batched_x.shape)  # (None, 3)
        print(batched_y.shape)  # (None, 3)
        ```

        Also automatically handles ragged batching
        ```python
        coords = b.prebatch_inputs(shape=(None, 3), dtype=tf.float32)
        batched_coords = b.batched(x)
        print(batched_coords)
        # tf.RaggedTensor(values=Tensor(shape=(3,)...), row_splits=...)
        ```

        Args:
            tensor: tensor structure

        Returns:
            batched tensor for each element of tensor.
        """
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
        """Marks the tensor as the input of the learned model."""
        if isinstance(tensor, tf.Tensor):
            return self._as_model_input(tensor)
        elif isinstance(tensor, tf.RaggedTensor):
            values = self.as_model_input(tensor.values)
            row_splits = self._as_model_input(tensor.row_splits)
            return tf.RaggedTensor.from_row_splits(values, row_splits)
        else:
            raise ValueError('Unrecognized type "%s"' % tensor)

    def as_batched_model_input(self, tensor):
        """Wrapper around `self.batched` -> `self.as_model_input`."""
        return self.as_model_input(self.batched(tensor))


# for use with context blocks
_base_builder = MetaNetworkBuilder()
_builder_stack = [_base_builder]


def prebatch_input(shape, dtype):
    """See `MetaNetworkBuilder.prebatch_input`."""
    return _builder_stack[-1].prebatch_input(shape=shape, dtype=dtype)


def prebatch_feed(tensor):
    """See `MetaNetworkBuilder.prebatch_feed`."""
    return _builder_stack[-1].prebatch_feed(tensor)


def prebatch_inputs_from(dataset):
    """See `MetaNetworkBuilder.prebatch_inputs_from`."""
    return _builder_stack[-1].prebatch_inputs_from(dataset)


def batched(tensor):
    """See `MetaNetworkBuilder.batched`."""
    return _builder_stack[-1].batched(tensor)


def as_model_input(tensor):
    """See `MetaNetworkBuilder.as_model_input`."""
    return _builder_stack[-1].as_model_input(tensor)


def as_batched_model_input(tensor):
    """See `MetaNetworkBuilder.as_batched_model_input`."""
    return _builder_stack[-1].as_batched_model_input(tensor)


def preprocessor():
    """See `MetaNetworkBuilder.preprocessor`."""
    return _builder_stack[-1].preprocessor()


def model(outputs):
    """See `MetaNetworkBuilder.model`."""
    return _builder_stack[-1].model(outputs)


def get_mark(tensor):
    """See `MetaNetworkBuilder.get_mark`."""
    return _builder_stack[-1].get_mark(tensor)
