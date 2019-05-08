from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections


RaggedComponents = collections.namedtuple(
    'RaggedComponents', ['flat_values', 'nested_row_splits'])


class Visitor(object):
    """Base class for recursively visiting tree structures."""
    def __init__(self, visit_fns=None, default_fn=None):
        self._visitors = {
            dict: self.visit_dict,
            tuple: self.visit_tuple,
            list: self.visit_list,
            # ListWrapper: self.visit_list_wrapper,
            tf.RaggedTensor: self.visit_ragged_tensor,
        }
        self._default_fn = default_fn
        if visit_fns is not None:
            self._visitors.update(visit_fns)

    def visit(self, obj):
        if obj is None:
            return self.visit_none()
        else:
            fn = self._visitors.get(type(obj))
            if fn is None:
                if isinstance(obj, list):
                    return self.visit_list(obj)
                if isinstance(obj, dict):
                    return self.visit_fn(obj)
                return self.visit_default(obj)

            return fn(obj)

    def visit_none(self):
        return None

    def visit_default(self, obj):
        if hasattr(obj, 'items'):
            return self.visit_dict(obj)
        return obj

    def visit_dict(self, obj):
        return {k: self.visit(v) for k, v in obj.items()}

    def visit_list(self, obj):
        return [self.visit(v) for v in obj]

    def visit_tuple(self, obj):
        return tuple(self.visit(v) for v in obj)

    def visit_ragged_tensor(self, obj):
        return obj

    def __call__(self, obj):
        return self.visit(obj)


class RaggedToComponents(Visitor):
    def visit_ragged_tensor(self, ragged_tensor):
        return RaggedComponents(
            self.visit(ragged_tensor.flat_values),
            [self.visit(splits) for splits in ragged_tensor.nested_row_splits])


class ComponentsToRagged(Visitor):
    def __init__(self):
        super(ComponentsToRagged, self).__init__({
            RaggedComponents: self.visit_ragged_components})

    def visit_ragged_components(self, obj):
        return tf.RaggedTensor.from_nested_row_splits(
            self.visit(obj.flat_values),
            [self.visit(splits) for splits in obj.nested_row_splits])


class ComponentsToNumpy(ComponentsToRagged):
    def visit_ragged_components(self, obj):
        from weighpoint.np_utils import ragged_array
        if len(obj.nested_row_splits) > 1:
            ragged_array.RaggedArray.from_nested_row_splits(
                self.visit(obj.flat_values), self.visit(obj.nested_row_splits))
        return ragged_array.RaggedArray.from_row_splits(
            self.visit(obj.flat_values), self.visit(obj.nested_row_splits[0]))


ragged_to_components = RaggedToComponents()
components_to_ragged = ComponentsToRagged()
components_to_numpy = ComponentsToNumpy()
