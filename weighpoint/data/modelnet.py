from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from absl import logging
import gin
import numpy as np
import tensorflow_datasets as tfds
import tensorflow_datasets.volume.modelnet.modelnet as mn
from weighpoint.data import problems
from weighpoint.data import augment
from weighpoint.np_utils import tree_utils
from weighpoint.np_utils import sample

import weighpoint.tf_compat  # noqa

SparseCategoricalCrossentropy = gin.config.external_configurable(
    tf.keras.losses.SparseCategoricalCrossentropy,
    name='SparseCategoricalCrossentropy'
)

DEFAULT_SEED = 123


@gin.configurable
class UniformUniformPointCloudModelnetConfig(mn.ModelnetConfig):
    """
    Point cloud config uniform cloud size and mean density.

    Clouds are NOT bound by a unit sphere. Rather, they are rescaled such that
    the mean `k`th neighbor is at `r0`.
    """
    def __init__(
            self, name='uniform_uniform_cloud_base', num_points=2048,
            k=20, r0=0.1, seed=DEFAULT_SEED, **kwargs):
        self._seed = seed
        self.r0 = r0
        self.k = k
        self.num_points = num_points
        super(UniformUniformPointCloudModelnetConfig, self).__init__(
          name=name,
          description='Uniform uniform point cloud',
          input_key='cloud', **kwargs)

    def input_features(self):
        return mn.cloud_features(2048, with_originals=True)

    def load(self, fp, path=None):
        from scipy.spatial import cKDTree
        import trimesh
        vertices, faces = mn.load_off_mesh(fp)
        # quick recentering to avoid numerical instabilities in sampling
        center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices -= center
        scale = np.max(vertices)
        vertices /= scale
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        positions, face_indices = trimesh.sample.sample_surface(
            mesh, self.num_points)
        normals = mesh.face_normals[face_indices]

        r, c = mn.naive_bounding_sphere(positions)
        positions -= c
        positions /= r
        center += c * scale
        scale *= r

        tree = cKDTree(positions)
        dists, indices = tree.query(tree.data, self.k)
        new_scale_factor = self.r0 / np.mean(dists[:, -1])
        positions *= new_scale_factor
        scale /= new_scale_factor

        return dict(
            positions=positions.astype(np.float32),
            normals=normals.astype(np.float32),
            original_center=center,
            original_radius=scale,
        )


@gin.configurable
class UniformDensityPointCloudModelnetConfig(mn.ModelnetConfig):
    def __init__(
          self, name='uniform_density_cloud_base',
          prob_scale_factor=2,
          num_points_start=2048,
          radius=0.1,
          sample_scale_factor=None,
          seed=DEFAULT_SEED,
          weight_fn=sample.gaussian,
          **kwargs):
        if sample_scale_factor is None:
            sample_scale_factor = radius / 2
        self._radius = radius
        self._sample_scale_factor = sample_scale_factor
        self._seed = seed
        self._num_points_start = num_points_start
        self._prob_scale_factor = prob_scale_factor
        self._weight_fn = weight_fn
        self._original_state = None
        super(UniformDensityPointCloudModelnetConfig, self).__init__(
          name=name,
          description='Uniform density point cloud',
          input_key='cloud', **kwargs)

    def input_features(self):
        return mn.cloud_features(None, with_originals=True)

    def load(self, fp, path=None):
        import trimesh
        from scipy.sparse.linalg import svds
        vertices, faces = mn.load_off_mesh(fp)
        # quick recentering to avoid numerical instabilities in sampling
        center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices -= center
        scale = np.max(vertices)
        vertices /= scale
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        positions, face_indices = trimesh.sample.sample_surface(
            mesh, self._num_points_start)
        normals = mesh.face_normals[face_indices]
        tree = tree_utils.cKDTree(positions)
        indices, neighborhood_size = tree_utils.query_ball_tree(
            tree, tree, self._radius)

        mask = sample.inverse_density_mask(
            neighborhood_size, prob_scale_factor=self._prob_scale_factor,
            indices=indices, coords=positions,
            scale_factor=self._sample_scale_factor, weight_fn=self._weight_fn)
        positions = positions[mask]
        if positions.shape[0] < 2:
            logging.info('Skipping %s' % path)
            self._bad_paths.append(path)
            return None
        normals = normals[mask]

        r, c = mn.naive_bounding_sphere(positions)
        positions -= c
        positions /= r
        center += c * scale
        scale *= r

        _, _, v = svds(positions, k=1)
        v = v[0]
        theta = np.arctan2(v[1], v[0])
        positions, normals = augment.rotate_np(positions, normals, -theta)
        return dict(
            positions=positions.astype(np.float32),
            normals=normals.astype(np.float32),
            original_center=center,
            original_radius=scale,
        )

    def __enter__(self):
        if self._original_state is not None:
            raise RuntimeError(
                "Cannot nest PointCloudModelnetConfig contexts.")
        self._original_state = np.random.get_state()
        np.random.seed(self._seed)
        self._bad_paths = []
        return self

    def __exit__(self, *args, **kwargs):
        # restore original numpy random state
        assert(self._original_state is not None)
        np.random.set_state(self._original_state)
        if len(self._bath_paths) == 0:
            logging.info('No bad paths!')
        else:
            logging.info('bad paths: \n%s' % ('\n'.join(self._bad_paths)))
            self._bad_paths = None
        self._original_state = None


@gin.configurable
def get_uniform_density_builder():
    return mn.Modelnet40(config=UniformDensityPointCloudModelnetConfig())


@gin.configurable
def get_uniform_size_builder(num_points=1024):
    if num_points != 1024:
        raise NotImplementedError()
    return tfds.builder('modelnet40/cloud1024')


@gin.configurable
def get_uniform_uniform_builder():
    return mn.Modelnet40(config=UniformUniformPointCloudModelnetConfig())


@gin.configurable
class UniformSampledConfig(mn.ModelnetSampledConfig):
    def __init__(self, num_classes=40, r0=0.1, k=20, num_sample_points=2048):
        super(UniformSampledConfig, self).__init__(
            num_classes, name_prefix='us')
        self.r0 = r0
        self.k = k
        self.num_sample_points = num_sample_points
        self._R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)

    def map_cloud(self, cloud):
        from scipy.spatial import cKDTree
        # Rotate so z is up
        positions = np.matmul(cloud['positions'], self._R)
        normals = np.matmul(cloud['normals'], self._R)
        r, c = mn.naive_bounding_sphere(positions)
        positions -= c
        sampled_positions = np.random.choice(
            np.array(range(len(positions)), dtype=np.int64),
            self.num_sample_points, replace=False)
        sampled_positions = positions[sampled_positions]
        tree = cKDTree(sampled_positions)
        dists, indices = tree.query(tree.data, self.k)
        recip_scale_factor = self.r0 / np.mean(dists[:, -1])
        positions *= recip_scale_factor
        return dict(
            positions=positions,
            normals=normals,
            scale_factor=np.array(1./recip_scale_factor, dtype=np.float32)
        )

    def input_features(self):
        num_points = self.num_points
        features = {
            "positions": tfds.features.Tensor(
                shape=(num_points, 3), dtype=tf.float32),
            "normals": tfds.features.Tensor(
                shape=(num_points, 3), dtype=tf.float32),
            "scale_factor": tfds.features.Tensor(shape=(), dtype=tf.float32)
        }
        return tfds.features.FeaturesDict(features)


@gin.configurable
def get_uus_builder():
    return mn.ModelnetSampled(config=UniformSampledConfig(40))


@gin.configurable
class ModelnetProblem(problems.TfdsProblem):
    def __init__(
            self, tfds_builder,
            loss=None,
            metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),),
            map_fn=None, as_supervised=True,
            shuffle_buffer=1024, download_and_prepare=True,
            min_points=256, alt_split_percent=None):
        if loss is None:
            loss = SparseCategoricalCrossentropy(from_logits=True)
        self._min_points = min_points
        self._alt_split_percent = alt_split_percent
        super(ModelnetProblem, self).__init__(
            builder=tfds_builder, loss=loss, metrics=metrics,
            map_fn=map_fn, as_supervised=as_supervised,
            shuffle_buffer=shuffle_buffer,
            download_and_prepare=download_and_prepare,
        )

    def _split(self, split):
        if self._alt_split_percent is None:
            return super(ModelnetProblem, self)._split(split)
        total = tfds.Split.TRAIN + tfds.Split.TEST
        if split == 'train':
            return total.subsplit(tfds.percent[self._alt_split_percent:])
        elif split == 'validation':
            return total.subsplit(tfds.percent[:self._alt_split_percent])
        else:
            raise NotImplementedError

    def examples_per_epoch(self, split='train'):
        def base(split):
            return int(self.builder.info.splits[split].num_examples)

        if self._alt_split_percent is None:
            return base('test' if split == 'validation' else split)

        total = sum(base(s) for s in ('train', 'test'))
        frac = float(self._alt_split_percent) / 100
        if split == 'train':
            return int((1 - frac)*total)
        elif split == 'validation':
            return int(frac*total)
        else:
            raise NotImplementedError

    def data_pipeline(self, dataset, split, batch_size, prefetch=True):
        if self._min_points is not None and self._min_points > 0:
            dataset = dataset.filter(
                lambda i, l: tf.greater_equal(
                    tf.shape(i['positions'])[0], self._min_points))
        map_fn = self._map_fn
        if isinstance(map_fn, dict):
            map_fn = map_fn[split]
        # if split == tfds.Split.TRAIN:
        dataset = dataset.shuffle(self._shuffle_buffer)
        if map_fn is None:
            def map_fn(inputs, labels):
                return {k: inputs[k] for k in ('positions', 'normals')}, labels

        dataset = dataset.map(map_fn, tf.data.experimental.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        if prefetch:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
