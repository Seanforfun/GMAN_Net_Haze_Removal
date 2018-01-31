#  ====================================================
#   Filename: dehazenet_input.py
#   Function: This file is used to read Clear and hazed images.
#   In the program, we read images and put them into related
#   Arrays.
#  ====================================================

import tensorflow as tf
import numpy as np

import os
import sys

from Image import *
import dehazenet as dehazenet


IMAGE_INDEX_BIT = 4


def image_list_shuffle(image_list):
    np.random.shuffle(image_list)
    return image_list

def image_input(dir, file_names, image_list, clear_dict, clear_image):
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
            image_input(os.path.join(dir, image_filename), file_names, image_list, clear_dict, clear_image)
        elif image_filename.endswith(".png") \
                | image_filename.endswith(".jpg"):
            file_name = os.path.join(dir, image_filename)
            file_names.append(file_name)
            current_image = Image(path=file_name)
            current_image.key = id(current_image)
            image_list.append(current_image)
            # Put all clear images into dictionary
            if clear_image:
                if len(image_filename) < IMAGE_INDEX_BIT + 4:
                    raise RuntimeError("Incorrect image name: " + image_filename)
                image_index = image_filename[0:IMAGE_INDEX_BIT]
                clear_dict[image_index] = current_image
    return file_names, image_list, clear_dict


def read_image(image_list, file_names=None):
    """
    :param image_list: A image list which saves the image objects
    :param file_names: A file name list(Optional)
    :return: A image list whose image_matrix is filled
    """
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


def get_distorted_image(image_batch_list, height, width, file_names=None):
    """
    :param batch_list: A list used to save a batch of image objects
    :param height: The height of our training image
    :param width: The width of our training image
    :param file_names: A batch of images to be trained(Optional)
    :return: A batch list of Image object whose image_matrix are filled
    :function: Used to read and distort a batch of images
    """
    if isinstance(height, int) | isinstance(width, int) \
            | image_batch_list is not None:
        for image in image_batch_list:
            if not tf.gfile.Exists(image.path):
                raise ValueError('Failed to find image: ' + image.path)
        image_batch = []
        #image_batch_list
        image_batch = read_image(image_batch_list)
        # TODO Image pre-processing
        for image in image_batch:
            if image.image_matrix is None:
                raise RuntimeError("Failed to read image: " + image.path)
            image.image_matrix = tf.random_crop(
                image.image_matrix, [dehazenet.FLAGS.input_image_height,
                                     dehazenet.FLAGS.input_image_width, 3])
            image.image_matrix = tf.image.random_brightness(image.image_matrix,
                                                         max_delta=63)
            image.image_matrix = tf.image.random_contrast(image.image_matrix,
                                                       lower=0.2, upper=1.8)
        return image_batch
    else:
        raise RuntimeError('Error input of method distorted_image')


if __name__ == '__main__':
    a = []
    # np.random.shuffle(a)
    # print(a)
    names = []
    dict = {}
    list=[]
    list, t, dict = image_input("./ClearImages", names, list, clear_dict=None,clear_image=False)
    print(t)
    image_list_shuffle(t)
    print(t)
