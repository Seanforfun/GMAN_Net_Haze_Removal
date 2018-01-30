#  ====================================================
#   Filename: dehazenet.py
#   Function:
#  ====================================================
import tensorflow as tf

import dehazenet_input as di

import os
import re
import sys

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('input_image_height', 128,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 128,
                            """Input image width.""")
tf.app.flags.DEFINE_string('haze_test_images_dir', './HazeImages/TestImages',
                           """Path to the hazed test images directory.""")
tf.app.flags.DEFINE_string('haze_train_images_dir', './HazeImages/TrainImages',
                           """Path to the hazed train images directory.""")
tf.app.flags.DEFINE_string('clear_train_images_dir', './ClearImages/TrainImages',
                           """Path to the clear train images directory.""")
tf.app.flags.DEFINE_string('clear_result_images_dir', './ClearImages/ResultImages',
                           """Path to the clear result images directory.""")
