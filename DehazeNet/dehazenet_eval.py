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
import dehazenet_flags as df
import numpy as np
import skimage.io as io
from skimage import transform
from PIL import Image as im

import re
from datetime import datetime
import os.path
import time

EVAL_TOWER_NAME = "tower"
EVAL_MOVING_AVERAGE_DECAY = 0.9999
# Frames used to save clear training image information
_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}
# Frames used to save hazed training image information
_hazed_test_file_names = []
_hazed_test_img_list = []
IMAGE_JPG_FORMAT = 'jpg'
IMAGE_PNG_FORMAT = 'png'

@DeprecationWarning
def convert_to_tfrecord(hazed_image_list, height, width):
    print('Start converting data into tfrecords...')
    writer = tf.python_io.TFRecordWriter(df.FLAGS.tfrecord_eval_path)
    for image in hazed_image_list:
        try:
            if not tf.gfile.Exists(image.path):
                raise ValueError("Image does not exist: " + image.path)
            hazed_image = im.open(image.path)
            reshape_hazed_image = hazed_image.resize((height, width))
            reshape_hazed_image_arr = np.array(reshape_hazed_image)
            hazed_image_raw = reshape_hazed_image_arr.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'hazed_image_raw': di.bytes_feature(hazed_image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            raise RuntimeError('Could not read:', image.path)
    writer.close();
    print('Transform done!')


def _eval_generate_image_batch(hazed_image, min_queue_examples, batch_size, shuffle=True):
    num_preprocess_threads = 8

    if shuffle:
        h_images, c_images = tf.train.shuffle_batch(
            [hazed_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        h_images, c_images = tf.train.batch(
            [hazed_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # Display the training images in the visualizer.
    tf.summary.image('hazed_images', h_images)
    return h_images, c_images

@DeprecationWarning
def read_eval_tfrecords_and_add_2_queue(tfrecords_filename, batch_size, height, width):
    if not tf.gfile.Exists(tfrecords_filename):
        raise ValueError("Fail to load TFRecord from dictionary: " + tfrecords_filename)
    filename_queue = tf.train.string_input_producer([tfrecords_filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'hazed_image_raw': tf.FixedLenFeature([], tf.string),
        })
    hazed_image = tf.decode_raw(img_features['hazed_image_raw'], tf.uint8)
    hazed_image = tf.reshape(hazed_image, [height, width, 3])
    hazed_image = tf.image.per_image_standardization(hazed_image)
    '''
        I am titanium.
    '''
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(di.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                             min_fraction_of_examples_in_queue)
    return _eval_generate_image_batch(hazed_image, min_queue_examples, batch_size, shuffle=True)


def evaluate():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 1.Create TFRecord for evaluate data
        if df.FLAGS.tfrecord_eval_rewrite:
            # 1.1 Read images from directory and save to memory
            di.image_input(df.FLAGS.haze_test_images_dir, _hazed_test_file_names, _hazed_test_img_list,
                           clear_dict=None, clear_image=False)
            if len(_hazed_test_img_list) == 0:
                raise RuntimeError("No image found! Please supply hazed images for eval ")
            di.image_input(df.FLAGS.clear_test_images_dir, _clear_test_file_names, _hazed_test_img_list,
                           clear_dict=_clear_test_directory, clear_image=True)
            # 1.2 Save images into TFRecord
            di.convert_to_tfrecord(_hazed_test_img_list, _hazed_test_file_names, _clear_test_directory,
                                   df.FLAGS.input_image_height, df.FLAGS.input_image_width, df.FLAGS.tfrecord_eval_path)
        # 2.Read data from TFRecord
        hazed_image, clear_image = di.read_tfrecords_and_add_2_queue(df.FLAGS.tfrecord_eval_path, df.FLAGS.batch_size,
                                                                     df.FLAGS.input_image_height,
                                                                     df.FLAGS.input_image_width)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([hazed_image, clear_image], capacity=2 * df.FLAGS.num_gpus)

        # 3.Dequeue in every GPU and create clear images
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(df.FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (dn.TOWER_NAME, i)) as scope:
                        hazed_image_batch, clear_image_batch = batch_queue.dequeue()
                        # 3.1 Train a batch of image and get a tensor used to represent the images
                        ground_truth_image_tensor = tf.squeeze(clear_image_batch, [0])
                        logist = dmgt.inference(hazed_image_batch)
                        predict_image_tensor = tf.squeeze(logist, [0])
                        image_name_base = str(time.time())
                        if df.FLAGS.save_image_type == IMAGE_JPG_FORMAT:
                            predict_image_jpg = tf.image.encode_jpeg(predict_image_tensor, format='rgb')
                            gt_image_jpg = tf.image.encode_jpeg(ground_truth_image_tensor, format='rgb')
                            with tf.gfile.GFile(df.FLAGS.clear_test_images_dir + image_name_base+'_pred.jpg',
                                                'wb') as f:
                                f.write(predict_image_jpg.eval())
                            with tf.gfile.GFile(df.FLAGS.clear_test_images_dir + image_name_base + '_gt.jpg',
                                                'wb') as f:
                                f.write(gt_image_jpg.eval())
                        elif df.FLAGS.save_image_type == IMAGE_PNG_FORMAT:
                            predict_image_jpg = tf.image.encode_png(predict_image_tensor, format='rgb')
                            gt_image_jpg = tf.image.encode_png(ground_truth_image_tensor, format='rgb')
                            with tf.gfile.GFile(df.FLAGS.clear_test_images_dir + image_name_base+'_pred.png',
                                                'wb') as f:
                                f.write(predict_image_jpg.eval())
                            with tf.gfile.GFile(df.FLAGS.clear_test_images_dir + image_name_base + '_gt.png',
                                                'wb') as f:
                                f.write(gt_image_jpg.eval())



def main():
    if df.FLAGS.tfrecord_eval_rewrite:
        if tf.gfile.Exists(df.FLAGS.tfrecord_eval_path):
            tf.gfile.Remove(df.FLAGS.tfrecord_eval_path)
            print('We delete the old TFRecord and will generate a new one in the program.')
    evaluate()


if __name__ == '__main__':
    tf.app.run()
