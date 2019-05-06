from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import six
import numpy as np
import tensorflow as tf
import gin


def factorial(n):
    return np.prod(range(1, n+1))


def get_geometric_polynomials(x, order):
    orders = tf.range(order, dtype=tf.float32)
    return tf.expand_dims(x, axis=-1) ** orders


class PolynomialBuilder(object):
    def get_polynomials(self, x, order):
        raise NotImplementedError('Abstract method')

    def __call__(self, x, order):
        return self.get_polynomials(x, order)


@gin.configurable
class GeometricPolynomialBuilder(PolynomialBuilder):
    def get_polynomials(self, x, order):
        return tf.unstack(get_geometric_polynomials(x, order), axis=-1)

    def __repr__(self):
        return 'GeomPolyBuilder'


class OrthogonalPolynomialBuilder(PolynomialBuilder):
    def get_polynomials(self, x, order):
        raise NotImplementedError('Abstract method')

    def get_domain(self):
        raise NotImplementedError('Abstract method')

    def get_normalization_factor(self, order):
        return 1

    def get_weighting_fn(self, x):
        return tf.ones_like(x)


class RecursiveOrthogonalPolynomialBuilder(OrthogonalPolynomialBuilder):
    def get_p0(self, x):
        return tf.ones_like(x)

    def get_p1(self, x):
        return x

    def get_next(self, x, pn2, pn1, n):
        raise NotImplementedError('Abstract method')

    def get_polynomials(self, x, order):
        if order < 0:
            raise ValueError('Order must be non-negative')
        p0 = self.get_p0(x)
        if order == 1:
            return [p0]
        p1 = self.get_p1(x)
        ps = [p0, p1]
        for n in range(2, order):
            ps.append(self.get_next(x, ps[n-2], ps[n-1], n))
        return ps


@gin.configurable
class LegendrePolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):
    def get_next(self, x, pn2, pn1, n):
        n -= 1
        return (2*n + 1) / (n + 1) * x * pn1 - n / (n+1) * pn2

    def get_normalization_factor(self, order):
        return 2 / (2*order + 1)

    def get_domain(self):
        return (-1, 1)

        def __repr__(self):
            return 'LegendrePolyBuilder'


class ChebyshevPolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):
    def get_next(self, x, pn2, pn1, n):
        return 2*x*pn1 - pn2

    def get_domain(self):
        return (-1, 1)

    @staticmethod
    def from_kind(kind='first'):
        if kind == 'first':
            return FirstChebyshevPolynomialBuilder()
        elif kind == 'second':
            return SecondChebyshevPolynomialBuilder()
        else:
            raise ValueError('`kind` must be one of "first", "second"')


@gin.configurable
class FirstChebyshevPolynomialBuilder(ChebyshevPolynomialBuilder):
    def get_p1(self, x):
        return x

    def get_weighting_fn(self, x):
        return 1 / tf.sqrt(1 - x**2)

    def get_normalization_factor(self, order):
        return np.pi if order == 0 else np.pi / 2

    def __repr__(self):
        return 'Chebyshev1PolyBuilder'


@gin.configurable
class SecondChebyshevPolynomialBuilder(ChebyshevPolynomialBuilder):
    def get_p1(self, x):
        return 2*x

    def get_weighting_fn(self, x):
        return tf.sqrt(1 - tf.square(x))

    def get_normalization_factor(self, order):
        return np.pi / 2

    def __repr__(self):
        return 'Chebyshev2PolyBuilder'


@gin.configurable
class HermitePolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):
    def get_p1(self, x):
        return 2*x

    def get_next(self, x, pn2, pn1, n):
        return 2*x*pn1 - 2*(n-1)*pn2

    def get_domain(self):
        return (-np.inf, np.inf)

    def get_weighting_fn(self, x):
        return tf.exp(-tf.square(x))

    def get_normalization_factor(self, order):
        return factorial(order) * 2**order * np.sqrt(np.pi)

    def __repr__(self):
        return 'HermitePolyBuilder'


@gin.configurable
class GaussianHermitePolynomialBuilder(OrthogonalPolynomialBuilder):
    def __init__(self, stddev=1.0):
        self.stddev = stddev

    def get_polynomials(self, x, order):
        hermites = HermitePolynomialBuilder().get_polynomials(
            x/self.stddev, order)
        f = np.sqrt(np.pi)*self.stddev
        exp_denom = 2*tf.square(self.stddev)
        for n, h in enumerate(hermites):
            if n > 0:
                f *= 2*n
            scale_factor = tf.exp(-tf.square(x)/exp_denom)/np.sqrt(f)
            hermites[n] = scale_factor*h
        return hermites

    def get_domain(self):
        return (-np.inf, np.inf)

    def get_normalization_factor(self, order):
        return 1

    def get_weighting_fn(self, x):
        return tf.ones_like(x)

    def __repr__(self):
        return 'GaussHermitePolyBuilder(%s)' % str(self.stddev).rstrip('0')


@gin.configurable
class GegenbauerPolynomialBuilder(RecursiveOrthogonalPolynomialBuilder):
    def __init__(self, lam=0.75):
        self.lam = lam

    def get_p1(self, x):
        return 2*x if self.lam == 0 else 2*self.lam * x

    def get_next(self, x, pn2, pn1, n):
        if n == 2 and self.lam == 0:
            return x*pn1 - 1
        else:
            rhs = 2*(n - 1 + self.lam)*x*pn1 - (n - 2 + 2*self.lam)*pn2
            return rhs / n

    def get_domain(self):
        return (-1, 1)

    def get_weighting_fn(self, x):
        return (1 - x**2)**(self.lam - 0.5)

    def get_normalization_factor(self, order):
        if self.lam == 0:
            if order == 0:
                return np.pi
            else:
                return 2*np.pi / order**2
        else:
            from scipy.special import gamma
            numer = np.pi*2**(1 - 2*self.lam)*gamma(order + 2*self.lam)
            denom = (order + self.lam) * factorial(order) *\
                gamma(self.lam)**2
            return numer / denom

    def __repr__(self):
        return 'GegenbauerPolyBuilder(%s)' % str(self.lam).rstrip('0')


def _total_order_num_out(num_dims, max_order):
    if num_dims == 0 or max_order == 0:
        return 1
    else:
        return sum(
            _total_order_num_out(num_dims-1, max_order-i)
            for i in range(max_order+1))


class NdPolynomialBuilder(object):
    def __init__(self, max_order=3, is_total_order=True, base_builder=None):
        if base_builder is None:
            base_builder = GeometricPolynomialBuilder()
        assert(callable(base_builder))
        self._base_builder = base_builder
        self._max_order = max_order
        self._is_total_order = is_total_order

    def num_out(self, num_dims):
        if self._is_total_order:
            return _total_order_num_out(num_dims, self._max_order) - 1
        else:
            return num_dims * self._max_order - 1

    def output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_out(input_shape[-1]),)

    def __call__(self, coords):
        if (
                self._max_order == 1 and
                self._is_total_order and
                isinstance(self._base_builder, GeometricPolynomialBuilder)):
            return coords
        single_polys = []
        coords = tf.unstack(coords, axis=-1)
        for i, x in enumerate(coords):
            polys = self._base_builder(x, self._max_order+1)
            single_polys.append(enumerate(polys))

        outputs = []
        for ordered_polys in itertools.product(*single_polys):
            orders, polys = zip(*ordered_polys)
            total_order = sum(orders)

            if total_order == 0 or (
                    self._is_total_order and total_order > self._max_order):
                continue
            outputs.append(tf.stack(polys, axis=-1))
        outputs = tf.stack(outputs, axis=-2)
        outputs = tf.reduce_prod(outputs, axis=-1)
        assert(outputs.shape[-1] == self.num_out(len(coords)))
        return outputs


_builder_factories = {
    'geo': GeometricPolynomialBuilder,
    'cheb': ChebyshevPolynomialBuilder.from_kind,
    'che1': FirstChebyshevPolynomialBuilder,
    'che2': SecondChebyshevPolynomialBuilder,
    'gh': GaussianHermitePolynomialBuilder,
    'her': HermitePolynomialBuilder,
    'geg': GegenbauerPolynomialBuilder,
    'leg': LegendrePolynomialBuilder,
}


def _builder_from_dict(name, **kwargs):
    return _builder_factories[name](**kwargs)


def deserialize_builder(obj):
    if obj is None:
        return None
    elif isinstance(obj, PolynomialBuilder):
        return obj
    elif isinstance(obj, six.string_types):
        return _builder_from_dict(obj)
    else:
        assert(isinstance(obj, dict))
        return _builder_from_dict(**obj)


def get_nd_polynomials(
        coords, max_order=3, is_total_order=True, base_builder=None):
    base_builder = deserialize_builder(base_builder)
    return NdPolynomialBuilder(max_order, is_total_order, base_builder)(coords)
