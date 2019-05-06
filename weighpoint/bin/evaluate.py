from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from weighpoint.bin.flags import parse_config
from weighpoint import runners


def evaluate(_):
    parse_config()
    runners.evaluate()


if __name__ == '__main__':
    app.run(evaluate)
