from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app
from absl import flags
from weighpoint.bin.flags import parse_config
from weighpoint import runners
from weighpoint.np_utils import vis

flags.DEFINE_string('split', 'train', '"train" or "test"')
flags.DEFINE_boolean('vis', True, 'whether or not to visualize')
flags.DEFINE_boolean('normalize', True, 'for visualization')
flags.DEFINE_boolean(
    'overwrite', False, 'if True, overwrites existing confusion matrices')


def confusion(_):
    parse_config()
    confs = []
    splits = ('train', 'validation')
    for split in splits:
        with tf.Graph().as_default():
            FLAGS = flags.FLAGS
            conf = runners.confusion(
                overwrite=FLAGS.overwrite, split=split)
            confs.append(conf)
    if FLAGS.vis:
        vis.plot_confusion_matrices(confs, splits, normalize=FLAGS.normalize)


if __name__ == '__main__':
    app.run(confusion)
