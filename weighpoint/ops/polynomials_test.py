from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
import weighpoint.ops.polynomials as p


def get_inner_products(builder, order, n_divs=1e6):
    n_divs = int(n_divs)
    a, b = builder.get_domain()

    def transform_bounds(a):
        if a == -np.inf:
            a = -100.0
        elif a == np.inf:
            a = 100.0
        elif isinstance(a, int):
            a = float(a)
        return a
    a = transform_bounds(a)
    b = transform_bounds(b)

    x = tf.linspace(a, b, n_divs)
    x = x[1:-1]
    polynomials = builder.get_polynomials(x, order)
    weighting_fn = builder.get_weighting_fn(x)

    norms = []
    dx = ((b-a) / n_divs)
    for i in range(order):
        ni = []
        poly = polynomials[i]
        integrand = poly*poly*weighting_fn
        integral = tf.reduce_sum(integrand) * dx
        ni.append(integral / builder.get_normalization_factor(i) - 1)
        for j in range(i+1, order):
            integrand = poly * polynomials[j] * weighting_fn
            integral = tf.reduce_sum(integrand) * dx
            ni.append(integral)
        norms.append(ni)
    return norms


@test_util.run_all_in_graph_and_eager_modes
class PolynomialBuilderTest(tf.test.TestCase):
    def _test_orthogonal(self, builder, order=5):
        with tf.device('cpu:0'):
            norms = get_inner_products(builder, order)
            norms = tf.concat(norms, axis=0)
            self.assertAllLess(self.evaluate(tf.abs(norms)), 1e-2)

    def test_legendre_orthogonal(self):
        self._test_orthogonal(p.LegendrePolynomialBuilder())

    def test_chebyshev_first_orthogonal(self):
        self._test_orthogonal(p.FirstChebyshevPolynomialBuilder())

    def test_chebyshev_second_orthogonal(self):
        self._test_orthogonal(p.SecondChebyshevPolynomialBuilder())

    def test_hermite_orthogonal(self):
        self._test_orthogonal(p.HermitePolynomialBuilder())

    def test_gaussian_hermite_orthogonal(self):
        for stddev in (1.0, 2.0):
            self._test_orthogonal(
                p.GaussianHermitePolynomialBuilder(stddev=stddev))

    def test_gegenbauer_orthogonal(self):
        for lam in (0.0, 0.75, 0.85):
            self._test_orthogonal(p.GegenbauerPolynomialBuilder(lam=lam))

    def test_nd(self):
        polys = p.NdPolynomialBuilder(
            max_order=3, is_total_order=True)(tf.random.normal(shape=(3,)))
        self.assertEqual(polys.shape.as_list(), [19])


if __name__ == '__main__':
    tf.test.main()
