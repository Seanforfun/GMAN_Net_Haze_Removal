#  ====================================================
#   Filename: dehazenet.py
#   Function: This file is entrance of the dehazenet.
#   Most of the parameters are defined in this file.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dehazenet_multi_gpu_train as dmgt
import dehazenet_flags as df


RGB_CHANNEL = 3

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def main(self):
    if tf.gfile.Exists(df.FLAGS.train_dir):
        tf.gfile.DeleteRecursively(df.FLAGS.train_dir)
    tf.gfile.MakeDirs(df.FLAGS.train_dir)
    print('start')
    dmgt.train()
    print('end')


if __name__ == '__main__':
    tf.app.run()