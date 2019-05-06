from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from weighpoint import visitor


def _actual_ragged_lambda_fn(
        flat_comp_inputs, fn, input_structure, cache, kwargs):
    if isinstance(flat_comp_inputs, tf.Tensor):
        flat_comp_inputs = [flat_comp_inputs]
    assert(isinstance(flat_comp_inputs, list))
    if not (all(isinstance(x, tf.Tensor) for x in flat_comp_inputs)):
        raise Exception(
            'Expected flat_inputs to be tensors, got %s'
            % str(flat_comp_inputs))
    inner_comp_inputs = tf.nest.pack_sequence_as(
        input_structure, flat_comp_inputs)
    args = visitor.components_to_ragged(inner_comp_inputs)
    output = fn(args, **kwargs)
    if isinstance(output, tf.Tensor):
        return output

    output = visitor.ragged_to_components(output)
    cache.append(tf.nest.map_structure(lambda x: True, output))
    flat_out = list(tf.nest.flatten(output))
    assert(type(flat_out) is list)
    return flat_out


def ragged_lambda_call(fn, args, arguments={}, name=None):
    assert(isinstance(arguments, dict))
    comp_inputs = visitor.ragged_to_components(args)
    flat_comp_inputs = list(tf.nest.flatten(comp_inputs))
    assert(all(isinstance(t, tf.Tensor) for t in flat_comp_inputs))
    structure = tf.nest.map_structure(lambda x: True, comp_inputs)

    cache = []
    flat_out = tf.keras.layers.Lambda(
        _actual_ragged_lambda_fn, arguments=dict(
            fn=fn, input_structure=structure, cache=cache, kwargs=arguments),
        name=name)(flat_comp_inputs)

    if isinstance(flat_out, tf.Tensor):
        return flat_out
    structure = cache.pop()
    assert(len(cache) == 0)
    out = tf.nest.pack_sequence_as(structure, flat_out)
    return visitor.components_to_ragged(out)


def ragged_lambda(fn, arguments={}, name=None):
    assert(isinstance(arguments, dict))

    def f(args):
        return ragged_lambda_call(fn, args, arguments, name)
    return f


def maybe_ragged_lambda_call(fn, args, arguments={}):
    if any(isinstance(x, tf.RaggedTensor) for x in tf.nest.flatten(args)):
        return ragged_lambda(fn, arguments=arguments)(args)
    else:
        return tf.keras.layers.Lambda(fn, arguments=arguments)(args)


def _ragged_from_tensor(args, **kwargs):
    tensor, lengths = args
    ragged = tf.RaggedTensor.from_tensor(tensor, lengths=lengths, **kwargs)
    return ragged


def ragged_from_tensor(tensor, lengths=None, **kwargs):
    if lengths is None:
        return ragged_lambda(
            tf.RaggedTensor.from_tensor, arguments=kwargs)(tensor)
    else:
        return ragged_lambda(_ragged_from_tensor, arguments=kwargs)(
            [tensor, lengths])


def _ragged_from_row_lengths(args):
    tensor, row_lengths = args
    return tf.RaggedTensor.from_row_lengths(tensor, row_lengths)


def ragged_from_row_lengths(tensor, row_lengths):
    return ragged_lambda(_ragged_from_row_lengths)([tensor, row_lengths])


def _ragged_from_nested_row_lengths(args):
    values = args[0]
    rls = args[1:]
    return tf.RaggedTensor.from_nested_row_lengths(values, rls)


def ragged_from_nested_row_lengths(flat_values, nested_row_lengths):
    return ragged_lambda(_ragged_from_nested_row_lengths)(
        [flat_values] + nested_row_lengths)


def _nested_row_lengths(rt):
    return list(rt.nested_row_lengths())


def nested_row_lengths(rt):
    assert(isinstance(rt, tf.RaggedTensor))
    out = ragged_lambda(_nested_row_lengths)(rt)
    if not isinstance(out, list):
        assert(not isinstance(out, tuple))
        out = [out]
    return out


def ragged_to_tensor(ragged, default_value=None):
    return ragged_lambda(
            tf.RaggedTensor.to_tensor,
            arguments=dict(default_value=default_value))(ragged)
