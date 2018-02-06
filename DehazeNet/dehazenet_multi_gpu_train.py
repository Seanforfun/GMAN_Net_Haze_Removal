#  ====================================================
#   Filename: dehazenet_multi_gpu_train.py
#   Function: This file defines the training function
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dehazenet_input as di
import dehazenet_flags as df
import dehazenet as dn
import dehazenet_tools as dt
import numpy as np
import re
from datetime import datetime
import os.path
import time


# Frames used to save clear training image information
_clear_train_file_names = []
_clear_train_img_list = []
_clear_train_directory = {}
# Frames used to save hazed training image information
_hazed_train_file_names = []
_hazed_train_img_list = []

# Some Strings
PROGRAM_START = "You shoot me down but I won't fall , I am titanium."
PROGRAM_END = "We don't talk anymore."


def inference(hazed_batch):
    """
    :param hazed_batch: The hazed training images from get_distorted_image.
    Each image is in the form of Images. 4D tensor of [batch_size, height, width, 3] size
    Please refer to CIFAR-10 CNN model to design our dehazenet.
    :return: A image batch after trained by CNN
    """
    # TODO Lida Xu please re-write the CNN model
    with tf.name_scope('DehazeNet'):
        x = dt.conv('conv1_1', hazed_batch, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv1_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # with tf.name_scope('pool1'):
        #     x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = dt.conv('conv2_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv2_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # with tf.name_scope('pool2'):
        #     x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = dt.conv('conv3_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv3_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv3_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # with tf.name_scope('pool3'):
        #     x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = dt.conv('conv4_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv4_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv4_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # with tf.name_scope('pool4'):
        #     x = tools.pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = dt.conv('conv5_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv5_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv5_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # with tf.name_scope('pool5'):
        #     x = tools.pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        x = dt.conv('conv6_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        x = dt.conv('conv6_2', x, 64, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.conv('conv6_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.conv('conv6_4', x, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        # x = tools.FC_layer('fc6', x, out_nodes=4096)
        # with tf.name_scope('batch_norm1'):
        #     x = tools.batch_norm(x)
        # x = tools.FC_layer('fc7', x, out_nodes=4096)
        # with tf.name_scope('batch_norm2'):
        #     x = tools.batch_norm(x)
        # x = tools.FC_layer('fc8', x, out_nodes=n_classes)

    return x


def loss(result_batch, clear_image_batch):
    """
    :param result_batch: A batch of image that been processed by out CNN
    :param clear_image_batch: The ground truth image to compare with result_batch
    :return: The loss value will be added to tensorflow graph, return is actually not necessary
    but is left here to show respect to CIFAR-10 source code
    """
    # TODO Lida Xu please redesign this function to achieve a better representation of loss
    loss = tf.reduce_mean(tf.square(tf.subtract(result_batch, clear_image_batch)))
    tf.add_to_collection('losses', loss)

    # The total loss is defined as the ms loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, hazed_batch, clear_batch):
    """Calculate the total loss on a single tower running the DeHazeNet model.

      Args:
        scope: unique prefix string identifying the DEHAZENET tower, e.g. 'tower_0'
        images: Images. 3D tensor of shape [height, width, 3].

      Returns:
         Tensor of shape [] containing the total loss for a batch of data
      """
    # Put our hazed images into designed CNN and get a result image batch
    logist = inference(hazed_batch)
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(logist, clear_batch)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % dn.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
    return total_loss, logist


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

     Note that this function provides a synchronization point across all towers.

     Args:
       tower_grads: List of lists of (gradient, variable) tuples. The outer list
         is over individual gradients. The inner list is over the gradient
         calculation for each tower.
     Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
     """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    print(PROGRAM_START)
    # Create all dehazenet information in /cpu:0
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # Calculate the learning rate schedule.
        num_batches_per_epoch = (di.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 df.FLAGS.batch_size)
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
        di.image_input(df.FLAGS.clear_train_images_dir, _clear_train_file_names, _clear_train_img_list,
                       _clear_train_directory, clear_image=True)
        if len(_clear_train_img_list) == 0:
            raise RuntimeError("No image found! Please supply clear images for training or eval ")
        # Hazed training image pre-process
        di.image_input(df.FLAGS.haze_train_images_dir, _hazed_train_file_names, _hazed_train_img_list,
                       clear_dict=None, clear_image=False)
        if len(_hazed_train_img_list) == 0:
            raise RuntimeError("No image found! Please supply hazed images for training or eval ")
        # Get queues for training image and ground truth, which is internally multi-thread safe
        hazed_image_queue, clear_image_queue = di.get_distorted_image(_hazed_train_img_list, df.FLAGS.input_image_height,
                                                                      df.FLAGS.input_image_width, _clear_train_directory,
                                                                      file_names=_hazed_train_file_names)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [hazed_image_queue, clear_image_queue], capacity=2 * df.FLAGS.num_gpus)
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(df.FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (dn.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU
                        hazed_image_batch, clear_image_batch = batch_queue.dequeue()
                        # Calculate the loss for one tower of the dehazenet model. This function
                        # constructs the entire dehazenet model but shares the variables across
                        # all towers.
                        loss, _ = tower_loss(scope, hazed_image_batch, clear_image_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            dn.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=df.FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(df.FLAGS.train_dir, sess.graph)

        for step in range(df.FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = df.FLAGS.batch_size * df.FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / df.FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == df.FLAGS.max_steps:
                checkpoint_path = os.path.join(df.FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    print(PROGRAM_END)


if __name__ == '__main__':
    tf.app.run()
