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
_clear_train_file_names = []
_clear_train_img_list = []
_clear_train_directory = {}
# Frames used to save hazed training image information
_hazed_train_file_names = []
_hazed_train_img_list = []


def _inference(hazed_batch):
    """
    :param hazed_batch: The hazed training images from get_distorted_image
    :return: A image batch after trained by CNN
    """
    # TODO Lida Xu please re-write the CNN model
    return 0


def _loss(result_batch, clear_image_batch):
    """
    :param result_batch: A batch of image that been processed by out CNN
    :param clear_image_batch: The ground truth image to compare with result_batch
    :return: The loss value will be added to tensorflow graph, return is actually not necessary
    but is left here to show respect to CIFAR-10 source code
    """
    # TODO Lida Xu please redesign this function to achieve a better representation of loss
    loss = tf.reduce_mean(tf.square(tf.subtract(result_batch, clear_image_batch)))
    tf.add_to_collection('losses', loss)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _tower_loss(scope, hazed_batch, clear_batch):
    """Calculate the total loss on a single tower running the DeHazeNet model.

      Args:
        scope: unique prefix string identifying the DEHAZENET tower, e.g. 'tower_0'
        images: Images. 3D tensor of shape [height, width, 3].

      Returns:
         Tensor of shape [] containing the total loss for a batch of data
      """
    # Put our hazed images into designed CNN and get a result image batch
    logist = _inference(hazed_batch)
    _ = _loss(logist, clear_batch)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


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
        di.image_input(dn.FLAGS.clear_train_images_dir, _clear_train_file_names, _clear_train_img_list,
                       _clear_train_directory, clear_image=True)
        # Hazed training image pre-process
        di.image_input(dn.FLAGS.haze_train_images_dir, _hazed_train_file_names, _hazed_train_img_list,
                       clear_dict=None, clear_image=False)
        # Get queues for training image and ground truth, which is internally multi-thread safe
        hazed_image_queue, clear_image_queue = di.get_distorted_image(_hazed_train_img_list, dn.FLAGS.input_image_height,
                                                                      dn.FLAGS.input_image_width)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [hazed_image_queue, clear_image_queue], capacity=2 * dn.FLAGS.num_gpus)

        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(dn.FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (dn.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU
                        hazed_image_batch, clear_image_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the dehazenet model. This function
                        # constructs the entire dehazenet model but shares the variables across
                        # all towers.
                        loss = _tower_loss(scope, hazed_image_batch, clear_image_batch)


if __name__ == '__main__':
    a = 10
    b = 3
    print(a/b)