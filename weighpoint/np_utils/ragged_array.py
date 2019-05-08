from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
# import scipy.sparse as sp


class RaggedArray(object):
    """
    Basic numpy equivalent of `tf.RaggedTensor`.

    This is purely to make a uniform interface for converting between different
    representations. Other than basic len/shape/iteration/__getitem__, no
    implementations of operations (e.g. addition, expand_dims etc) are
    included.
    """
    def __init__(
            self, values=None, row_lengths=None, ragged_lists=None,
            row_splits=None, csr_mat=None, leading_dim=None, internal=False):
        """
        For internal use only.

        Use one of the following:
            RaggedArray.from_ragged_list
            RaggedArray.from_row_lengths
            RaggedArray.from_row_splits
        """
        assert(internal)
        self._values = values
        self._row_lengths = row_lengths
        self._row_splits = row_splits
        self._ragged_lists = ragged_lists
        self._leading_dim = leading_dim

    @staticmethod
    def from_ragged_tensor_value(self, ragged_tensor_value):
        return RaggedArray.from_row_splits(
            ragged_tensor_value.values, ragged_tensor_value.row_splits)

    @staticmethod
    def from_ragged_lists(ragged_lists, dtype=np.float32):
        row_lengths = np.array(
            tuple(len(r) for r in ragged_lists), dtype=np.int64)
        size = np.sum(row_lengths)
        flat_values = np.empty((size,), dtype=dtype)
        ragged_lists = tuple(np.asarray(r, dtype=dtype) for r in ragged_lists)
        np.concatenate(ragged_lists, axis=0, out=flat_values)
        return RaggedArray(
            flat_values, row_lengths, ragged_lists=ragged_lists,
            leading_dim=row_lengths.size, internal=True)

    @staticmethod
    def from_row_lengths(values, row_lengths):
        values = np.asarray(values)
        row_lengths = np.asarray(row_lengths)
        return RaggedArray(
            values=values, row_lengths=row_lengths,
            leading_dim=row_lengths.size, internal=True)

    @staticmethod
    def from_nested_row_splits(values, nested_row_splits):
        for row_splits in nested_row_splits[-1::-1]:
            values = RaggedArray.from_row_splits(values, row_splits)
        return values

    @staticmethod
    def from_nested_row_lengths(values, nested_row_lengths):
        for row_lengths in nested_row_lengths[-1::-1]:
            values = RaggedArray.from_row_lengths(values, row_lengths)
        return values

    @staticmethod
    def from_row_splits(
            values, row_splits, values_dtype=None, splits_dtype=np.int64):
        if isinstance(values, (np.ndarray, list, tuple)):
            values = np.asarray(values, dtype=values_dtype)
        else:
            assert(isinstance(values, RaggedArray))
        row_splits = np.asarray(row_splits, dtype=splits_dtype)
        return RaggedArray(
            values=values, row_splits=row_splits,
            leading_dim=row_splits.size - 1, internal=True)

    @property
    def ragged_lists(self):
        if self._ragged_lists is None:
            values = self.values
            row_splits = self.row_splits
            if isinstance(values, RaggedArray):
                raise NotImplementedError()
            else:
                self._ragged_lists = np.split(values, row_splits[1:-1])
        return self._ragged_lists

    @property
    def row_splits(self):
        if self._row_splits is None:
            row_lengths = self.row_lengths
            row_splits = np.empty((row_lengths.size + 1,), dtype=np.int64)
            np.cumsum(row_lengths, out=row_splits[1:])
            row_splits[0] = 0
            self._row_splits = row_splits
        return self._row_splits

    @property
    def row_lengths(self):
        if self._row_lengths is None:
            if self._ragged_lists is None:
                self._row_lengths = np.diff(self._row_splits)
            else:
                self._row_lengths = np.array(
                    tuple(len(l) for l in self._ragged_lists), dtype=np.int64)

        return self._row_lengths

    @property
    def values(self):
        """Alias to be consistent with tensorflow."""
        return self._values

    @property
    def flat_values(self):
        values = self.values
        while isinstance(values, RaggedArray):
            values = values.values
        return values

    @property
    def size(self):
        return self.flat_values.size

    @property
    def leading_dim(self):
        return self._leading_dim

    @property
    def shape(self):
        return (self.leading_dim, None) + self._values.shape[1:]

    def __len__(self):
        return self.leading_dim

    def mask(self, boolean_mask):
        # assert(
        #     isinstance(boolean_mask, np.ndarray) and
        #     boolean_mask.dtype == np.bool)
        # indices = np.arange(self.size)
        # mat = sp.csr_matrix((self.values, indices, self.row_splits))
        # mat = mat[boolean_mask]
        # mat.sort_indices()
        # return RaggedArray.from_row_splits(mat.data, mat.indptr)
        ragged_lists = [
            r for r, m in zip(self.ragged_lists, boolean_mask) if m]
        return RaggedArray.from_ragged_lists(ragged_lists, dtype=self.dtype)

    def gather(self, indices):
        ragged_lists = self.ragged_lists
        ragged_lists = [ragged_lists[i] for i in indices]
        return RaggedArray.from_ragged_lists(ragged_lists, dtype=self.dtype)
        # indices = np.arange(self.size)
        # mat = sp.csr_matrix((self.values, indices, self.row_splits))
        # mat = mat[indices]
        # mat.sort_indices()
        # return RaggedArray.from_row_splits(mat.data, mat.indptr)

    def __getitem__(self, indices):
        if (isinstance(indices, int) or
                isinstance(indices, np.ndarray) and
                len(indices.shape) == 0 and
                np.issubdtype(indices.dtype, np.integer)):
            row_splits = self.row_splits
            i = indices
            return self.values[row_splits[i]: row_splits[i+1]]
        elif isinstance(indices, np.ndarray):
            if indices.dtype == np.bool:
                return self.mask(indices)
            elif np.issubdtype(indices.dtype, np.integer):
                return self.gather(indices)
            else:
                raise ValueError('Indices must be ')
        else:
            assert(isinstance(indices, tuple))
            i, j = indices[:2]
            start = self.row_splits[i]
            out = self._values[start: start + j]
            if len(indices) > 2:
                return out[indices[2:]]
            else:
                return out

    def __iter__(self):
        return iter(self[i] for i in range(self.leading_dim))

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def nested_row_splits(self):
        out = [self.row_splits]
        values = self.values
        while isinstance(values, RaggedArray):
            out.append(self.row_splits)
            values = values.values
        return out

    @property
    def nested_row_lengths(self):
        out = [self.row_lengths]
        values = self.values
        while isinstance(values, RaggedArray):
            out.append(self.row_lengths)
            values = values.values
        return out
