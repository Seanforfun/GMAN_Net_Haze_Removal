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


def convert_to_tfrecord(hazed_image_list):
    print('Start converting data into tfrecords...')
    writer = tf.python_io.TFRecordWriter(df.FLAGS.tfrecord_eval_path)
    for image in hazed_image_list:
        if not tf.gfile.Exists(image.path):
            raise ValueError("Image does not exist: " + image.path)
        hazed_image = im.open(image.path)
    pass


def _save_clear_image(path, clear_image_tensor):
    # TODO Write the clear image into specific path
    pass


def _evaluate_single_batch(hazed_test_image_batch, clear_test_image_batch, dest_dir):
    # TODO Restore our CNN from trained data
    # TODO Run operations and create the corresponding clear images
    pass


def evaluate():
    # 1.Create TFRecord for evaluate data
    if df.FLAGS.tfrecord_eval_rewrite:
        # 1.1 Read images from directory and save to memory
        di.image_input(df.FLAGS.haze_test_images_dir, _hazed_test_file_names, _hazed_test_img_list,
                       clear_dict=None, clear_image=False)
        if len(_hazed_test_img_list) == 0:
            raise RuntimeError("No image found! Please supply hazed images for training or eval ")
        # 1.2 Save images into TFRecord
        convert_to_tfrecord(_hazed_test_img_list)
    pass


def main():
    if df.FLAGS.tfrecord_eval_rewrite:
        if tf.gfile.Exists(df.FLAGS.tfrecord_eval_path):
            tf.gfile.Remove(df.FLAGS.eval_dir)
            print('We delete the old TFRecord and will generate a new one in the program.')
    evaluate()


if __name__ == '__main__':
    tf.app.run()
