from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import gin


def get_pca_xy_angle(positions):
    from sklearn.decomposition import PCA

    def get_pca(xy):
        pca = PCA(n_components=1)
        xy = xy.numpy()
        pca.fit_transform(xy)
        pca_vec = tf.squeeze(pca.components_, axis=0)
        return pca_vec

    xy, _ = tf.split(positions, [2, 1], axis=-1)
    pca_vec = tf.py_function(get_pca, [xy], positions.dtype)
    pca_vec.set_shape((2,))
    x, y = tf.unstack(pca_vec, axis=0)
    return tf.atan2(y, x)


def _rotate(positions, normals=None, angle=None, impl=tf):
    """
    Randomly rotate the point cloud about the z-axis.

    Args:
        positions: (n, 3) float array
        normals (optional): (n, 3) float array
        angle: float scalar. If None, a uniform random angle in [0, 2pi) is
            used.
        impl:

    Returns:
        rotated (`positions`, `normals`). `normals` will be None if not
        provided. shape and dtype is the same as provided.
    """
    dtype = positions.dtype
    if angle is None:
        angle = tf.random.uniform((), dtype=dtype) * (2 * np.pi)

    if normals is not None:
        assert(normals.dtype == dtype)
    c = impl.cos(angle)
    s = impl.sin(angle)
    # multiply on right, use non-standard rotation matrix (-s and s swapped)
    rotation_matrix = impl.reshape(
        impl.stack([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]),
        (3, 3))

    positions = impl.matmul(positions, rotation_matrix)
    if normals is None:
        return positions, None
    else:
        return positions, impl.matmul(normals, rotation_matrix)


@gin.configurable(whitelist=['angle'])
def rotate(positions, normals=None, angle=None):
    """See _rotate. `angle` may also be 'pca-xy'."""
    if angle == 'pca-xy':
        angle = get_pca_xy_angle(positions)
    return _rotate(positions, normals, angle, tf)


def rotate_np(positions, normals, angle):
    return _rotate(positions, normals, angle, np)


@gin.configurable(blacklist=['positions'])
def jitter_positions(positions, stddev=0.01, clip=0.05):
    """
    Randomly jitter points independantly by normally distributed noise.

    Args:
        positions: float array, any shape
        stddev: standard deviation of jitter
        clip: if not None, jittering is clipped to this
    """
    jitter = tf.random.normal(shape=tf.shape(positions), stddev=stddev)
    if clip is not None:
        jitter = tf.clip_by_norm(jitter, clip, axes=[-1])
    # scale by max norm
    max_val = tf.reduce_max(tf.linalg.norm(positions, axis=-1))
    jitter = jitter * max_val
    return positions + jitter


@gin.configurable(blacklist=['normals'])
def jitter_normals(
        normals, stddev=0.01, clip=0.05, some_normals_invalid=False):
    if stddev == 0:
        return normals
    normals = jitter_positions(normals, stddev, clip)
    norms = tf.linalg.norm(normals, axis=-1, keepdims=True)
    if some_normals_invalid:
        # some normals might be invalid, in which case they'll initially be 0.
        thresh = 0.1 if clip is None else 1 - clip
        return tf.where(
            tf.less(norms, thresh), tf.zeros_like(normals), normals / norms)
    else:
        return normals / norms


def reflect_x(xyz):
    x, y, z = tf.unstack(xyz, axis=-1)
    return tf.stack([-x, y, z], axis=-1)


@gin.configurable(blacklist=['inputs', 'labels'])
def augment_cloud(
        inputs, labels,
        rotate=rotate,
        jitter_positions=jitter_positions,
        jitter_normals=jitter_normals,
        scale_range=None,
        maybe_reflect_x=False,
        sample_prob=None, take_first_n=None):
    positions = inputs['positions']
    normals = inputs['normals']
    if rotate is not None:
        positions, normals = rotate(positions, normals)
    if scale_range is not None:
        min_scale, max_scale = scale_range
        scale = min_scale + (max_scale - min_scale)*tf.random.uniform(shape=())
        positions = positions * scale
    if maybe_reflect_x:
        should_reflect = tf.random.uniform(shape=(), dtype=tf.float32) > 0.5
        positions, normals = tf.cond(
            should_reflect,
            lambda: (reflect_x(positions), reflect_x(normals)),
            lambda: (positions, normals))

    if jitter_positions is not None:
        positions = jitter_positions(positions)
    if jitter_normals is not None:
        normals = jitter_normals(normals)
    if sample_prob is not None:
        if isinstance(sample_prob, tuple):
            minv, maxv = sample_prob
            sample_prob = minv + tf.random.uniform(shape=())*(maxv - minv)
        r = tf.random.uniform(
            shape=(tf.shape(positions)[0],), dtype=tf.float32)
        mask = tf.less(r, sample_prob)
        positions = tf.boolean_mask(positions, mask)
        normals = tf.boolean_mask(normals, mask)
    if take_first_n is not None:
        if isinstance(take_first_n, float):
            take_first_n = tf.cast(
                take_first_n * tf.shape(positions)[0], tf.int64)
        positions = positions[:take_first_n]
        normals = normals[:take_first_n]
    inputs = dict(positions=positions, normals=normals)
    return inputs, labels
