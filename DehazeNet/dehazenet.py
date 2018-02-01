#  ====================================================
#   Filename: dehazenet.py
#   Function: This file is entrance of the dehazenet.
#   Most of the parameters are defined in this file.
#  ====================================================
import tensorflow as tf
import dehazenet_input as di
import dehazenet_tools as dt
import dehazenet_eval as de
import dehazenet as dn
import numpy as np

import os
import re
import sys

FLAGS = tf.app.flags.FLAGS
RGB_CHANNEL = 3;
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = di.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('input_image_height', 128,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 128,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('original_height', 100000,
                            """Input image original height.""")
tf.app.flags.DEFINE_integer('original_width', 100000,
                            """Input image original width.""")
tf.app.flags.DEFINE_string('haze_test_images_dir', './HazeImages/TestImages',
                           """Path to the hazed test images directory.""")
tf.app.flags.DEFINE_string('haze_train_images_dir', './HazeImages/TrainImages',
                           """Path to the hazed train images directory.""")
tf.app.flags.DEFINE_string('clear_train_images_dir', './ClearImages/TrainImages',
                           """Path to the clear train images directory.""")
tf.app.flags.DEFINE_string('clear_result_images_dir', './ClearImages/ResultImages',
                           """Path to the clear result images directory.""")


# Some systematic parameters
tf.app.flags.DEFINE_string('train_dir', './DeHazeNetEventLog',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('num_cpus', 1,
                            """How many CPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def main():
    pass


if __name__ == '__main__':
    tf.app.run()