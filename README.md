# WeighPoint: Weighted Point Cloud Convolutions

Point cloud convolutions in deep learning have seen a lot of interest lately. Broadly speaking, these involve grouping points according to local proximity using some data structure like a KDTree. While results on classification and segmentation tasks are promising, most publicly available implementations suffer from a number of factors including:

1. `k`-nearest-neighbors search to ensure a fixed number of neighbors, rather than a fixed neighborhood size (as is suggested should be the case for convolutions in integral form);
2. discontinuity in space: `k`-nearest neighbors is discontinous as the `k`th and `k+1`th neighbors switch order; and
3. custom kernels which require additional setup and maintenance.

To address these, we implement:

1. neighborhoods defined by ball-searches, implemented using `tf.RaggedTensor`s;
2. a *weighted convolution* operation that ensures the integrated function trails to zero at the ball-search radius and is invariant to point-density; and
3. a meta-network building process that allows per-layer preprocessing operations to be built in conjunction with the learnable operations before being split into separate preprocessing and learned `tf.keras.Model`s.

The resulting architecture makes efficient use of CPUs for preprocessing without the need for custom kernels. As a bonus, the radii over which the ball searches occur can also be learned.

The project is under heavy development. Currently we are able to achieve competitive results on modelnet40 classification task (~90%) and are working towards semantic segmentation and point cloud generation implementations.

## Usage

```bash
git clone https://github.com/jackd/weighpoint.git
cd weighpoint
pip install -r requirements.txt
pip install -e .
cd config/cls
python ../../weighpoint/bin/train.py --gin_file=uup-exp-geo1-ctg-l2-lrd2b
```

The first time running this will require the download and preprocessing of the modelnet dataset. This will take quite some time (20min-ish, depending on computer/internet connection). The progress bar may appear frozen at times, but this is due to issues partially reading `tar` files. It should resolve itself eventually.

Training progress can be observed using tensorboard:

```bash
cd ~/weighpoint/cls
tensorboard --logdir=./
```

The first epoch or two will normally have terrible evaluation loss/accuracy. This is due to fast training in the early stages outpacing the batch-normalization statistics, resulting in out-of-sync offset/scalings in evaluation mode.

## Theory

See [this paper](https://drive.google.com/open?id=1VxAnRMcPhovMwqpY1Z9iYsOS8pjvdLFw) for a basic overview of the theory associated with the operations.

## Python/Tensorflow Compatibility

While we provide no guarantees, best efforts have been made to make this code compatible with python 2/3 and tensorflow versions `>=1.13` and `2.0.0-alpha`. Please note `weighpoint.tf_compat` includes some very dirty monkey-patching to make everything feel closer to `2.0`. Importing any `weighpoint` subpackages will result in changes to the `tf` namespace and sub-name spaces which may affect external code.

## Neighborhood Implementations

We use [`KDTree.query_pairs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) (along with random sampling and/or masking) to calculate neighborhoods with a variable number of neighbors. While no tensorflow implementation exists, we find performance is acceptable using `tf.function` during preprocessing (i.e. inside a `tf.data.Dataset.map`).

We store the calculated neighborhoods in [`tf.RaggedTensor`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/RaggedTensor?hl=en)s where possible and make extensive use of `tf.ragged` operations.

## Data Pipeline

The data pipeline developed in this project is critical to the timely training of the networks without introducing custom operations. It is made up of:

* [tensorflow_dataset](github.com/jackd/tensorflow/datasets) base implementations that manage downloading and serialization of the raw point cloud data;
* [weighpoint/data/augmentation](./weighpoint/data/augment) for model-independent preprocessing;
* [weighpoint/meta](./weighpoint/meta) for tools to write per-layer preprocessing operations (like KDTrees) along with learned components. The result is a learnable `tf.keras.Model` along with a model-dependent preprocessing and batching functions with a `tf.data.Dataset` pipeline.

## Classification

TODO

## Segmentation

Work ongoing. Known bug in `tree_utils` associated with `reverse_query_ball`.

## Point Cloud Generation

Work ongoing
