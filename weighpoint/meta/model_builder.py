from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from weighpoint.meta import utils as meta_utils


def mark(tensor):
    get_model_builder(tensor).mark(tensor)


def get_model_builder(tensor):
    builder = getattr(tensor, '_builder', None)
    if builder is None:
        for dep in tensor.op.inputs:
            builder = get_model_builder(tensor)
            if builder is not None:
                tensor._builder = builder
                break
    return builder


def mark_output(tensor):
    get_model_builder(tensor).mark_output(tensor)


def dataset_map_inputs(builder, dataset):
    output_types = dataset.output_types
    types = tf.nest.flatten(output_types)
    names = tuple(
        '-'.join(str(s) for s in p) for p in
        meta_utils.yield_flat_paths(output_types))
    shapes = tf.nest.flatten(dataset.output_shapes)
    inputs = [builder.input(s, d, n) for s, d, n in zip(shapes, types, names)]
    return tf.nest.pack_sequence_as(output_types, inputs)


def repack_outputs(requested_outputs, flat_outputs_dict):
    def f(output):
        builder = output._builder
        if builder not in flat_outputs_dict:
            raise KeyError(
                'No flat_outputs provided for builder %s' % builder.name)
        index = builder._output_indices[output]
        return flat_outputs_dict[builder][index]

    return tf.nest.map_structure(f, requested_outputs)


class ModelBuilder(meta_utils.Finalizable):
    def __init__(self, name):
        self._inputs = []
        self._outputs = []
        self._output_indices = {}
        self._name = name
        self._model = None

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self.name

    def __repr__(self):
        return 'ModelBuilder(%s)' % self.name

    @property
    def finalized(self):
        return self._model is None

    def finalize(self):
        if self.finalized:
            return
        self._model = tf.keras.models.Model(
            inputs=self._inputs, outputs=self._outputs)

    def _input(self, shape, dtype, name=None):
        self._assert_mutable('add new input')
        if name is None:
            name = 'input%d' % len(self._inputs)
        inp = tf.keras.layers.Input(
            shape=shape, dtype=dtype, batch_size=1,
            name='%s-%s' % (self._name, name))
        self._inputs.append(inp)
        self.mark(inp)
        return inp

    def input(self, shape, dtype, name=None):
        inp = self._input(shape=shape, dtype=dtype, name=name)
        inp = tf.keras.layers.Lambda(tf.squeeze, arguments=dict(axis=0))(inp)
        return inp

    def as_input(self, tensor, name=None):
        if not isinstance(tensor, tf.Tensor):
            raise ValueError('Only tensors can be used as inputs')
        mark_output(tensor)
        inp = self._input(tensor.shape, tensor.dtype, name=name)
        self._input_map[inp] = tensor
        inp = tf.keras.layers.Lambda(tf.squeeze, arguments=dict(axis=0))(inp)
        return inp

    def mark(self, tensor):
        """
        Mark tensor and all dependencies as belonging to this builder.

        Raises:
            ValueError: if tensor or any dependencies are marked by another
                `ModelBuilder`.
            RuntimeError: if executing eagerly.
        """
        self._assert_mutable('mark')
        if tf.executing_eagerly():
            raise RuntimeError('`ModelBuidler.mark` requires graph mode')
        mark = getattr(tensor, '_builder', None)
        if mark is None:
            tensor._mark = self
            for dep in tensor.op.inputs:
                self.mark(dep)
        elif mark is not self:
            raise ValueError(
                'Cannot mark tensor - already marked by %s' % tensor._builder)

    def mark_output(self, tensor):
        self._assert_mutable('mark output')
        if tensor in self._output_indices:
            return
        self.mark(tensor)
        self._output_indices[tensor] = len(self._outputs)
        self._outputs.append(tensor)

    def call_model(self, flat_input_values):
        self._assert_finalized('call model')
        flat_input_tensors = [
            tf.keras.layers.Lambda(tf.expand_dims, arguments=dict(axis=0))(t)
            for t in flat_input_values]
        return self.model(flat_input_tensors)
