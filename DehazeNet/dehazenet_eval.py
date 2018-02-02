#  ====================================================
#   Filename: dehazenet_eval.py
#   Function: This file is used for evaluate our model and create a
#   image from a hazed image.
#  ====================================================

import tensorflow as tf
import dehazenet_input as di
import dehazenet_tools as dt
import dehazenet_eval as de
import dehazenet as dn


# Frames used to save clear training image information
_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}
# Frames used to save hazed training image information
_hazed_test_file_names = []
_hazed_test_img_list = []


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './DeHazeNetEval',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('clear_test_images_dir', './ClearImages/TestImages',
                           """Path to the clear result images directory.""")
tf.app.flags.DEFINE_string('haze_test_images_dir', './HazeImages/TestImages',
                           """Path to the hazed test images directory.""")


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
    with tf.Graph().as_default() as g:
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

    # TODO STEP2:Use inference to create a test operation
    # TODO STEP3:Call _evaluate_single_batch() to run the test program
    pass


def main():
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if '__name__' == '__main__':
    tf.app.run()
