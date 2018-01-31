#  ====================================================
#   Filename: dehazenet_input.py
#   Function: This file is used to read Clear and hazed images.
#   In the program, we read images and put them into related
#   Arrays.
#  ====================================================

import tensorflow as tf
import numpy as np

import os

from Image import *
import dehazenet as dehazenet

def image_input(dir, file_names, image_list):
    """
        :param dir: The directory to read the image
        :param file_names: An empty list to save all image file names
        :return: A list used to save all Image objects.A list used to save names
    """
    if not dir:
        raise ValueError('Please supply a data_dir')
    file_list = os.listdir(dir)
    for image_filename in file_list:
        if os.path.isdir(os.path.join(dir, image_filename)):
            image_input(os.path.join(dir, image_filename), file_names, image_list)
        elif image_filename.endswith(".png") \
                | image_filename.endswith(".jpg"):
            file_name = os.path.join(dir, image_filename)
            file_names.append(file_name)
            current_image = Image(path=file_name)
            current_image.key = id(current_image)
            print(current_image.key)
            image_list.append(current_image)
    return file_names, image_list


def read_picture(image_list, file_names=None):
    """
    :param filename_queue: A queue used to save all file names
    :param image_list:A list to save all image objects
    :return: A single image object
    """
    # record_bytes = dehazenet.FLAGS.original_height \
    #                * dehazenet.FLAGS.original_width * dehazenet.RGB_CHANNEL
    # reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # key, value = reader.read(filename_queue)
    # current_image = Image(key=key);
    # depth_major = tf.reshape(value, [dehazenet.FLAGS.original_height,
    #                                  dehazenet.FLAGS.original_width,
    #                                  dehazenet.RGB_CHANNEL])
    # current_image.image_matrix = depth_major
    for image in image_list:
        image_content = tf.read_file(image.path)
        if image.path.endswith(".png"):
            image_matrix = tf.image.decode_png(image_content,
                                        channels=dehazenet.RGB_CHANNEL, dtype=tf.uint8)
        elif image.path.endswith("jpg"):
            image_matrix = tf.image.decode_jpeg(image_content, channels=dehazenet.RGB_CHANNEL)
        image_matrix = tf.cast(image_matrix, tf.float32)
        rshape = tf.reshape(tf.reduce_mean(image_matrix, [0, 1]), [1, 1, 3])
        image_matrix = image_matrix / rshape * 128
        image.image_matrix = image_matrix
    return image_list


def distorted_image(image_list, file_names, height, width):
    """
    :param image_list: A list used to save all image objects
    :param file_names: A batch of images to be trained
    :param height: The height of our training image
    :param width: The width of our training image
    :return: A batch list of Image object
    """
    if isinstance(height, int) | isinstance(width, int) \
            | file_names is not None:
        for image in file_names:
            if not tf.gfile.Exists(image.path):
                raise ValueError('Failed to find image: ' + image.path)
        image_list = read_picture(image_list)

        # TODO Image pre-processing
        for image in image_list:
            image.image_matrix = tf.random_crop(
                image.image_matrix, [dehazenet.FLAGS.input_image_height,
                                     dehazenet.FLAGS.input_image_width, 3])
            image.image_matrix = tf.image.random_brightness(image.image_matrix,
                                                         max_delta=63)
            image.image_matrix = tf.image.random_contrast(image.image_matrix,
                                                       lower=0.2, upper=1.8)
        # TODO Put all graphs into queue
    else:
        raise RuntimeError('Error input of method distorted_image')


if __name__ == '__main__':
    a = []
    list = []
    names = []
    list, t = image_input("./testd", names, list)
    print(t)
