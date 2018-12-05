#  ====================================================
#   Filename: gman_train.py
#   Function: This file defines the training function
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from datetime import datetime

import gman_config as dc
import gman_constant as constant
import gman_input as di
import gman_log as logger
import gman_model as model
import gman_net as net
import gman_tower as tower
import gman_learningrate as learning_rate
from PerceNet import *


def train_load_previous_model(path, saver, sess, init=None):
    gmean_ckpt = tf.train.get_checkpoint_state(path)
    if gmean_ckpt and gmean_ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, gmean_ckpt.model_checkpoint_path)
    else:
        sess.run(init)


def train(tf_record_path, image_number, config):
    logger.info("Training on: %s" % tf_record_path)
    tf.reset_default_graph()
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

        initial_learning_rate = learning_rate.LearningRate(constant.INITIAL_LEARNING_RATE, df.FLAGS.train_learning_rate)
        lr = tf.train.exponential_decay(initial_learning_rate.load(),
                                        global_step,
                                        decay_steps,
                                        initial_learning_rate.decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(lr)
        # opt = tf.train.GradientDescentOptimizer(lr)

        batch_queue = di.input_get_queue_from_tfrecord(tf_record_path, df.FLAGS.batch_size,
                                                       df.FLAGS.input_image_height, df.FLAGS.input_image_width)
        # Calculate the gradients for each model tower.
        # vgg_per = Vgg16()
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            gman_model = model.GMAN_V1()
            gman_net = net.Net(gman_model)
            for i in range(df.FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (constant.TOWER_NAME, i)) as scope:
                        gman_tower = tower.GMEAN_Tower(gman_net, batch_queue, scope, tower_grads, opt)
                        summaries, loss = gman_tower.process()

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = tower.Tower.average_gradients(tower_grads)
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
            log_device_placement=df.FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=constant.TRAIN_GPU_MEMORY_ALLOW_GROWTH,
                                      per_process_gpu_memory_fraction=constant.TRAIN_GPU_MEMORY_FRACTION,
                                      visible_device_list=constant.TRAIN_VISIBLE_GPU_LIST))
        )

        # Restore previous trained model
        if config[dc.CONFIG_TRAINING_TRAIN_RESTORE]:
            train_load_previous_model(df.FLAGS.train_dir, saver, sess)
        else:
            sess.run(init)

        coord = tf.train.Coordinator()
        # Start the queue runners.
        queue_runners = tf.train.start_queue_runners(sess=sess, coord=coord, daemon=False)

        summary_writer = tf.summary.FileWriter(df.FLAGS.train_dir, sess.graph)
        max_step = int((image_number / df.FLAGS.batch_size) * 2)
        # For each tf-record, we train them twice.
        for step in range(max_step):
            start_time = time.time()
            if step != 0 and (step % 1000 == 0 or (step + 1) == max_step):
                _, loss_value, current_learning_rate = sess.run([train_op, loss, lr])
            else:
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

            if step % 1000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step != 0 and (step % 1000 == 0 or (step + 1) == max_step):
                checkpoint_path = os.path.join(df.FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                initial_learning_rate.save(current_learning_rate)

        coord.request_stop()
        sess.close()
        coord.join(queue_runners, stop_grace_period_secs=constant.TRAIN_STOP_GRACE_PERIOD, ignore_live_threads=True)
    logger.info("=========================================================================================")


if __name__ == '__main__':
    pass
