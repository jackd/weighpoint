from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import gin

flags.DEFINE_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS


def parse_config():
    import weighpoint.models.core  # noqa
    gin_file = FLAGS.gin_file
    if not gin_file.endswith('.gin'):
        gin_file = '%s.gin' % gin_file
    gin.bind_parameter('model_dir.model_id', gin_file[:-4])
    gin.parse_config_files_and_bindings([gin_file], FLAGS.gin_param)
