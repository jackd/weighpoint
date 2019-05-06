from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from weighpoint.tf_compat import dim_value


def default_padding_value(dtype):
    return False if dtype.is_bool else tf.constant(-1, dtype)


def needs_padding(dataset):
    return any(
        len(shape) > 0 and dim_value(shape[0]) is None for shape in
        tf.nest.flatten(dataset.output_shapes))


def batch_dataset(dataset, batch_size, padding_values=None, **kwargs):
    # padded batch is any output first dims are None.
    if needs_padding(dataset):
        return dataset.padded_batch(
            batch_size,
            padded_shapes=dataset.output_shapes,
            padding_values=padding_values,
            **kwargs)
    else:
        return dataset.batch(batch_size, **kwargs)


class Preprocessor(object):
    def __init__(self, prebatch_model, prebatch_feed_values, batched_model):
        self._prebatch_model = prebatch_model
        self._batched_model = batched_model
        self._prebatch_feed_values = prebatch_feed_values

        for m, name in (
                (self._prebatch_model, 'prebatch'),
                (self._batched_model, 'batched')):
            if len(m.trainable_variables) > 0:
                raise RuntimeError(
                    'Expected no trainable variables in %s model but got %s' %
                    (name, str(m.trainable_variables)))

    @classmethod
    def from_io(
            cls,
            prebatch_inputs,
            prebatch_outputs,
            prebatch_feed_inputs,
            prebatch_feed_values,
            batched_inputs,
            batched_outputs,
            ):
        prebatch_model = tf.keras.models.Model(
            inputs=prebatch_inputs + prebatch_feed_inputs,
            outputs=prebatch_outputs)
        batched_model = tf.keras.models.Model(
            inputs=batched_inputs, outputs=batched_outputs)
        return Preprocessor(
            prebatch_model, prebatch_feed_values, batched_model)

    def map_prebatch_inputs(self, inputs):
        inputs = (inputs, self._prebatch_feed_values)
        inputs = tf.nest.map_structure(
            lambda a: tf.expand_dims(a, axis=0), inputs)
        inputs = tf.nest.flatten(inputs)
        outputs = self._prebatch_model(inputs)
        return (outputs,) if isinstance(outputs, tf.Tensor) else tuple(outputs)

    def prebatch_map(self, inputs, labels):
        return self.map_prebatch_inputs(inputs), labels

    def map_batched_inputs(self, inputs):
        inputs = tf.nest.flatten(inputs)
        outputs = self._batched_model(inputs)
        return (outputs,) if isinstance(outputs, tf.Tensor) else tuple(outputs)

    def postbatch_map(self, inputs, labels):
        return self.map_batched_inputs(inputs), labels

    def map_and_batch(
            self, dataset, batch_size, num_parallel_calls=None,
            padding_value_fn=default_padding_value, batch_kwargs={}):
        dataset = dataset.map(
            self.prebatch_map, num_parallel_calls=num_parallel_calls)
        padding_values = tf.nest.map_structure(
            padding_value_fn, dataset.output_types)
        dataset = batch_dataset(
            dataset, batch_size=batch_size, padding_values=padding_values,
            **batch_kwargs)
        dataset = dataset.map(
            self.postbatch_map, num_parallel_calls=num_parallel_calls)
        return dataset
