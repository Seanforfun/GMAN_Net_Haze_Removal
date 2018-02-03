#  ====================================================
#   Filename: dehazenet_eval.py
#   Function: This file is used for evaluate our model and create a
#   image from a hazed image.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dehazenet_input as di
import dehazenet_tools as dt
import dehazenet_eval as de
import dehazenet_multi_gpu_train as dmgt
import dehazenet as dn


# Frames used to save clear training image information
_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}
# Frames used to save hazed training image information
_hazed_test_file_names = []
_hazed_test_img_list = []


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './DeHazeNet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './DeHazeNetEval',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('clear_test_images_dir', './ClearImages/TestImages',
                           """Path to the clear result images directory.""")
tf.app.flags.DEFINE_string('haze_test_images_dir', './HazeImages/TestImages',
                           """Path to the hazed test images directory.""")
tf.app.flags.DEFINE_string('clear_result_images_dir', './ClearResultImages',
                           """Path to the hazed test images directory.""")


def _save_clear_image(path, clear_image_tensor):
    # TODO Write the clear image into specific path
    pass


def _evaluate_single_batch(hazed_test_image_batch, clear_test_image_batch, dest_dir):
    # TODO Restore our CNN from trained data
    # TODO Run operations and create the corresponding clear images
    pass


def evaluate():
    # TODO Mengzhen Wang please re-write this function to achieve a better performance
    # This function read the model from ./DeHazeNetModel/model.ckpt
    # file and dehaze the hazed test images. This final result graph will
    # be put into ./ClearImages/ResultImages

    # TODO STEP1:Get a test hazed and clear image batch from queue, which is not shuffled
    with tf.Graph().as_default() as g, tf.device('/cpu:0'):
        # Create a variable to count the number of evaluate() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # Read test data and pre-process
        # Clear training image pre-process
        di.image_input(dn.FLAGS.clear_train_images_dir, _clear_test_file_names, _clear_test_img_list,
                       _clear_test_directory, clear_image=True)
        if len(_clear_test_img_list) == 0:
            raise RuntimeError("No image found! Please supply clear images for training or eval ")
        # Hazed training image pre-process
        di.image_input(dn.FLAGS.haze_train_images_dir, _hazed_test_file_names, _hazed_test_img_list,
                       clear_dict=None, clear_image=False)
        if len(_hazed_test_img_list) == 0:
            raise RuntimeError("No image found! Please supply hazed images for training or eval ")

        #Get image queues
        hazed_image_queue, clear_image_queue = di.get_distorted_image(_hazed_test_img_list,
                                                                      dn.FLAGS.input_image_height,
                                                                      dn.FLAGS.input_image_width,
                                                                      _clear_test_directory, Train=False)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [hazed_image_queue, clear_image_queue], capacity=2 * dn.FLAGS.num_gpus)

        global_loss = []
        # Calculate the gradients for each model tower.
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(dn.FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (dn.TOWER_NAME, i)) as scope:
                        # Dequeues one batch for the GPU
                        hazed_image_batch, clear_image_batch = batch_queue.dequeue()

                        # Build a Graph that computes the logits predictions from the
                        # inference model.
                        # TODO STEP2:Use inference to create a test operation
                        result_image_batch = dmgt.inference(hazed_image_batch)

                        losses = dmgt.loss(result_image_batch, clear_image_batch)

                        # Write the clear image into specific path
                        _save_clear_image(FLAGS.clear_result_images_dir, result_image_batch)

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            dn.MOVING_AVERAGE_DECAY, global_step)

        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            # TODO Need to define a directory to save log
            log_device_placement=dn.FLAGS.log_device_placement))

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

    # TODO STEP3:Create a program used for printing test result
    pass


def main():
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if '__name__' == '__main__':
    tf.app.run()
