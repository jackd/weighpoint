from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import app
from weighpoint.bin.flags import parse_config
from weighpoint import runners


def train(_):
    with tf.Graph().as_default():
        parse_config()
        runners.train()


if __name__ == '__main__':
    app.run(train)
