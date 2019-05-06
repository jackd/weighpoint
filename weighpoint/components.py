from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


TensorComponents = collections.namedtuple(
    'TensorComponents', ['tensor', 'lengths'])

RowLengthComponents = collections.namedtuple(
    'RowLengthComponents', ['values', 'row_lengths'])

RowSplitComponents = collections.namedtuple(
    'RowSplitComponents', ['values', 'row_splits'])
