#  ====================================================
#   Filename: dehazenet_multi_gpu_train.py
#   Function: This file defines the training function
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
from datetime import datetime

import dehazenet_constant as constant
import dehazenet_input as di
import dehazenet_tools as dt
import dehazenet_log as logger
from PerceNet import *

# Frames used to save clear training image information
_clear_train_file_names = []
_clear_train_img_list = []
_clear_train_directory = {}
# Frames used to save hazed training image information
_hazed_train_file_names = []
_hazed_train_img_list = []

_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}


def inference(hazed_batch):
    """
    :param hazed_batch: The hazed training images from get_distorted_image.
    Each image is in the form of Images. 4D tensor of [batch_size, height, width, 3] size
    Please refer to CIFAR-10 CNN model to design our dehazenet.
    :return: A image batch after trained by CNN
    """
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

    return x


def lz_training_net(hazed_batch):
    x_s = dt.conv('DN_conv1_1', hazed_batch, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv1_2', x_s, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x = dt.conv('upsampling_1', x, 64, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
    x = dt.conv('upsampling_2', x, 128, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])

    x1 = dt.conv('DN_conv1_3', x, 128, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv2_1', x1, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti('DN_conv2_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x1)
    # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    # x = tools.conv('DN_conv2_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv2_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x2 = dt.conv('DN_conv3_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv3_2', x2, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti('DN_conv3_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x2)
    # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv3_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x3 = dt.conv('DN_conv4_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv4_2', x3, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv4_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti('DN_conv4_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x3)
    # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    x = dt.conv('DN_conv4_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x4 = dt.conv('DN_conv5_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv5_2', x4, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv5_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv('DN_conv5_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti('DN_conv5_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x4)
    x = dt.acti_layer(x)

    x = dt.deconv('DN_deconv1', x, 64, 64, output_shape=[35, 112, 112, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])

    x = dt.deconv('DN_deconv2', x, 64, 64, output_shape=[35, 224, 224, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
    x = dt.conv('DN_conv6_6', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti('DN_conv6_7', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x_s)
    # # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    x = dt.conv('DN_conv6_8', x, 64, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    return x


def loss(result_batch, clear_image_batch):
    """
    :param result_batch: A batch of image that been processed by out CNN
    :param clear_image_batch: The ground truth image to compare with result_batch
    :return: The loss value will be added to tensorflow graph, return is actually not necessary
    but is left here to show respect to CIFAR-10 source code
    """

    # output_per_1, output_per_2, output_per_3 = vgg_per.build(result_batch)
    # output_tru_1, output_tru_2, output_tru_3 = vgg_per.build(clear_image_batch)
    # vgg_tru = Vgg16()
    # vgg_tru.build(clear_image_batch)

    # output_per_1 = vgg_per.conv3_3
    # output_tru_1 = vgg_tru.conv3_3
    #
    # output_per_2 = vgg_per.conv1_1
    # output_tru_2 = vgg_tru.conv1_1
    #
    # output_per_3 = vgg_per.conv2_2
    # output_tru_3 = vgg_tru.conv2_2

    # per_loss = (tf.reduce_mean(tf.square(tf.subtract(output_per_1, output_tru_1))) / 3136) + \
    #            (tf.reduce_mean(tf.square(tf.subtract(output_per_2, output_tru_2))) / 50176) + \
    #            (tf.reduce_mean(tf.square(tf.subtract(output_per_3, output_tru_3))) / 12544)
    loss = tf.reduce_mean(tf.square(tf.subtract(result_batch, clear_image_batch)))# + 0.01 * per_loss
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
    logist = lz_training_net(hazed_batch)
    # logist = inference(hazed_batch)
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
        loss_name = re.sub('%s_[0-9]*/' % constant.TOWER_NAME, '', l.op.name)
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


def train(tf_record_path, image_number):
    logger.info("Training on: %s" % tf_record_path)
    # Create all dehazenet information in /cpu:0
    with tf.Graph().as_default():
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # Calculate the learning rate schedule.
        if constant.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN < df.FLAGS.batch_size:
            raise RuntimeError(' NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN cannot smaller than batch_size!')
        num_batches_per_epoch = (constant.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 df.FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * constant.NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(constant.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        constant.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)
        # opt = tf.train.GradientDescentOptimizer(lr)

        # Image pre-process
        if df.FLAGS.tfrecord_rewrite:
            di.image_input(df.FLAGS.clear_test_images_dir, _clear_test_file_names, _clear_test_img_list,
                           clear_dict=_clear_test_directory, clear_image=True)
            di.image_input(df.FLAGS.clear_train_images_dir, _clear_train_file_names, _clear_train_img_list,
                           _clear_train_directory, clear_image=True)
            if len(_clear_train_img_list) == 0:
                raise RuntimeError("No image found! Please supply clear images for training or eval ")
            # Hazed training image pre-process
            di.image_input(df.FLAGS.haze_train_images_dir, _hazed_train_file_names, _hazed_train_img_list,
                           clear_dict=None, clear_image=False)
            if len(_hazed_train_img_list) == 0:
                 raise RuntimeError("No image found! Please supply hazed images for training or eval ")

            # Write data into a TFRecord saved in path ./TFRecord
            di.convert_to_tfrecord(_hazed_train_img_list, _hazed_train_file_names, _clear_train_directory,
                                   df.FLAGS.input_image_height, df.FLAGS.input_image_width, df.FLAGS.tfrecord_path, _clear_test_img_list)
        hazed_image, clear_image = di.read_tfrecords_and_add_2_queue(tf_record_path, df.FLAGS.batch_size,
                                                                     df.FLAGS.input_image_height, df.FLAGS.input_image_width)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [hazed_image, clear_image], capacity=2 * df.FLAGS.num_gpus)
        # Calculate the gradients for each model tower.
        # vgg_per = Vgg16()
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(df.FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (constant.TOWER_NAME, i)) as scope:
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
        variable_averages = tf.train.ExponentialMovingAverage(constant.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        # , variables_averages_op
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

        # Restore previous trained model
        if df.FLAGS.train_restore:
            dehazenet_ckpt = tf.train.get_checkpoint_state(df.FLAGS.train_dir)
            if dehazenet_ckpt and dehazenet_ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, dehazenet_ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        # Start the queue runners.
        queue_runners = tf.train.start_queue_runners(sess=sess, coord=coord, daemon=False)

        summary_writer = tf.summary.FileWriter(df.FLAGS.train_dir, sess.graph)

        for step in range(image_number / df.FLAGS.batch_size):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = df.FLAGS.batch_size * df.FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / df.FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.8f (%.1f examples/sec; %.3f '
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

        coord.request_stop()
        sess.close()
        coord.join(queue_runners, stop_grace_period_secs=constant.TRAIN_STOP_GRACE_PERIOD, ignore_live_threads=True)
    print(constant.PROGRAM_END)


if __name__ == '__main__':
    pass
