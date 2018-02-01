#  ====================================================
#   Filename: dehazenet_multi_gpu_train.py
#   Function: This file defines the training function
#  ====================================================

import tensorflow as tf
import dehazenet_input as di
import dehazenet_tools as dt
import dehazenet_eval as de
import dehazenet as dn
import numpy as np


# Frames used to save clear training image information
__clear_train_file_names = []
__clear_train_img_list = []
__clear_train_directory = {}
# Frames used to save hazed training image information
__hazed_train_file_names = []
__hazed_train_img_list = []


def _inference():
    pass


def _tower_loss(scope, images):
    """Calculate the total loss on a single tower running the DeHazeNet model.

      Args:
        scope: unique prefix string identifying the DEHAZENET tower, e.g. 'tower_0'
        images: Images. 3D tensor of shape [height, width, 3].

      Returns:
         Tensor of shape [] containing the total loss for a batch of data
      """
    _inference(images)
    pass


def train():
    # Create all dehazenet information in /cpu:0
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # Calculate the learning rate schedule.
        num_batches_per_epoch = (dn.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 dn.FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * dn.NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(dn.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        dn.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        # Image pre-process
        # Clear training image pre-process
        di.image_input(dn.FLAGS.clear_train_images_dir, __clear_train_file_names, __clear_train_img_list, __clear_train_directory, clear_image=True)
        # Hazed training image pre-process and shuffle
        di.image_input(dn.FLAGS.haze_train_images_dir, __hazed_train_file_names, __hazed_train_img_list,
                       clear_dict=None, clear_image=False)
        _ = di.image_list_shuffle(__hazed_train_img_list)

        # Dequeue from the image list


if __name__ == '__main__':
    a = 10
    b = 3
    print(a/b)