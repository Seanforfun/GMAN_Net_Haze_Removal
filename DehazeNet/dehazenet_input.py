#  ====================================================
#   Filename: dehazenet_input.py
#   Function: This file is used to read Clear and hazed images.
#   In the program, we read images and put them into related
#   Arrays.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import os
import skimage.io as io
from skimage import transform
from PIL import Image as im

from Image import *
import dehazenet as dehazenet
import dehazenet_flags as df
# import dehazenet_darkchannel as dd


IMAGE_INDEX_BIT = 4
# TODO Need to change value before real operations
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 69965
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
IMAGE_SUFFIX_MIN_LENGTH = 4


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
                | image_filename.endswith(".jpg") | image_filename.endswith(".bmp"):
            file_name = os.path.join(dir, image_filename)
            current_image = Image(path=file_name)
            current_image.key = id(current_image)
            current_image.image_index = image_filename[0:IMAGE_INDEX_BIT]
            image_list.append(current_image)
            # Put all clear images into dictionary
            if clear_image:
                if len(image_filename) < IMAGE_INDEX_BIT + IMAGE_SUFFIX_MIN_LENGTH:
                    raise RuntimeError("Incorrect image name: " + image_filename)
                clear_dict[current_image.image_index] = current_image
    if not clear_image:
        image_list_shuffle(image_list)
    for image in image_list:
        file_names.append(image.path)
    return file_names, image_list, clear_dict

@DeprecationWarning
def _read_image(filename):
    """
    :param image_list: A image list which saves the image objects
    :param file_names: A file name list(Optional)
    :return: A image list whose image_tensor is filled
    """
    if filename is None:
        raise RuntimeError("Fail to load path from dictionary")
    image_content = tf.read_file(filename)
    if filename.endswith(".png"):
        tensor = tf.image.decode_png(image_content, channels=dehazenet.RGB_CHANNEL, dtype=tf.uint8)
    elif filename.endswith("jpg"):
        tensor = tf.image.decode_jpeg(image_content, channels=dehazenet.RGB_CHANNEL)
    return tensor


def _generate_image_batch(hazed_image, clear_image, min_queue_examples, batch_size, shuffle=True):
    num_preprocess_threads = 8

    if shuffle:
        h_images, c_images = tf.train.shuffle_batch(
            [hazed_image, clear_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        h_images, c_images = tf.train.batch(
            [hazed_image, clear_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # Display the training images in the visualizer.
    tf.summary.image('hazed_images', h_images)
    tf.summary.image('clear_images', c_images)
    return h_images, c_images

@DeprecationWarning
def _image_pre_process(image, height, width, train=False):
    image = tf.reshape(image, [height, width, 3])
    image = tf.cast(image, tf.float32)
    if train:
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(image)
    # Resize the input image
    # tf.image.resize_images(float_image, [height, width])
    # float_image.set_shape([height, width, 3])
    return float_image


def find_corres_clear_image(image, clear_dict):
    clear_image_obj = clear_dict[image.image_index]
    if not tf.gfile.Exists(clear_image_obj.path):
        raise RuntimeError("Fail to load path from dictionary: " + clear_image_obj.path)
    clear_image = im.open(clear_image_obj.path)
    return clear_image

@DeprecationWarning
def _find_corres_clear_image_filenames(hazed_image_list, clear_dict):
    clear_image_name_list = []
    for image in hazed_image_list:
        clear_image_obj = clear_dict[image.image_index]
        clear_image_name_list.append(clear_image_obj.path)
    return clear_image_name_list

@DeprecationWarning
def hazed_get_distorted_image(image_batch_list, height, width, Train=True, file_names=None):
    """
    :param image_batch_list: A list used to save a batch of image objects
    :param height: The height of our training image
    :param width: The width of our training image
    :param file_names: A batch of images to be trained(Optional)
    :return: A batch list of Image object whose image_tensor are filled
    :function: Used to read and distort a batch of images
    """
    if isinstance(height, int) & isinstance(width, int):
        for image in image_batch_list:
            if not tf.gfile.Exists(image.path):
                raise ValueError('Failed to find image from path: ' + image.path)
        filename_queue = tf.train.string_input_producer(file_names, shuffle=False)
        hazed_original_image = _read_image(filename_queue)
        reshape_hazed_image = _image_pre_process(hazed_original_image, height, width)
        resize_hazed = tf.image.resize_images(reshape_hazed_image, [height, width])
        min_fraction_of_examples_in_queue = 0.1
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        return _generate_image_batch(resize_hazed, min_queue_examples, df.FLAGS.batch_size,
                                     shuffle=False)
    else:
        raise RuntimeError('Error input of method distorted_image')

@DeprecationWarning
def clear_get_distorted_image(hazed_image_list, height, width, dict, Train=False, file_names=None):
    """
    :param image_batch_list: A list used to save a batch of image objects
    :param height: The height of our training image
    :param width: The width of our training image
    :param file_names: A batch of images to be trained(Optional)
    :return: A batch list of Image object whose image_tensor are filled
    :function: Used to read and distort a batch of images
    """
    if isinstance(height, int) & isinstance(width, int):
        for image in hazed_image_list:
            if not tf.gfile.Exists(image.image_index):
                raise ValueError('Failed to find image from path: ' + image.image_index)
        clear_image_names = _find_corres_clear_image_filenames(hazed_image_list, dict)
        clear_filename_queue = tf.train.string_input_producer(clear_image_names, shuffle=False)
        clear_original_image = _read_image(clear_filename_queue)
        reshape_clear_image = _image_pre_process(clear_original_image, height, width, train=False)
        resize_clear = tf.image.resize_images(reshape_clear_image, [height, width])
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        return _generate_image_batch(resize_clear, min_queue_examples, df.FLAGS.batch_size,
                                     shuffle=Train)
    else:
        raise RuntimeError('Error input of method distorted_image')


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(hazed_image_list, hazed_image_file_names, dict, height, width, tfrecord_path):
    if len(hazed_image_list) == 0:
        raise RuntimeError("No example found for training! Please check your training data set!")
    for image in hazed_image_list:
        if not tf.gfile.Exists(image.path):
            raise ValueError('Failed to find image from path: ' + image.path)
    print('Start converting data into tfrecords...')
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for image in hazed_image_list:
        try:
            hazed_image = im.open(image.path)
            shape = np.shape(hazed_image)
            if df.FLAGS.input_image_width % 2 != 0:
                left = df.FLAGS.input_image_width // 2
                right = left + 1
            else:
                left = df.FLAGS.input_image_width / 2
                right = left
            if df.FLAGS.input_image_height % 2 != 0:
                up = df.FLAGS.input_image_height // 2
                low = up + 1
            else:
                up = df.FLAGS.input_image_height / 2
                low = up
            reshape_hazed_image = hazed_image.crop(
                (shape[1] // 2 - left, shape[0] // 2 - up, shape[1] // 2 + right, shape[0] // 2 + low))
            # reshape_hazed_image = hazed_image.resize((height, width), resample=im.BICUBIC)
            reshape_hazed_image_arr = np.array(reshape_hazed_image)
            hazed_image_raw = reshape_hazed_image_arr.tostring()
            clear_image = find_corres_clear_image(image, dict)
            shape = np.shape(clear_image)
            if df.FLAGS.input_image_width % 2 != 0:
                left = df.FLAGS.input_image_width // 2
                right = left + 1
            else:
                left = df.FLAGS.input_image_width / 2
                right = left
            if df.FLAGS.input_image_height % 2 != 0:
                up = df.FLAGS.input_image_height // 2
                low = up + 1
            else:
                up = df.FLAGS.input_image_height / 2
                low = up
                reshape_clear_image = clear_image.crop(
                (shape[1] // 2 - left, shape[0] // 2 - up, shape[1] // 2 + right, shape[0] // 2 + low))
            # reshape_clear_image = clear_image.resize((height, width))
            reshape_clear_image_arr = np.array(reshape_clear_image)
            clear_image_raw = reshape_clear_image_arr.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'hazed_image_raw': bytes_feature(hazed_image_raw),
                'clear_image_raw': bytes_feature(clear_image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            raise RuntimeError('Could not read:', image.path)
    writer.close()
    print('Transform done!')


def read_tfrecords_and_add_2_queue(tfrecords_filename, batch_size, height, width):
    if not tf.gfile.Exists(tfrecords_filename):
        raise ValueError("Fail to load TFRecord from dictionary: " + tfrecords_filename)
    filename_queue = tf.train.string_input_producer([tfrecords_filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'hazed_image_raw': tf.FixedLenFeature([], tf.string),
            'clear_image_raw': tf.FixedLenFeature([], tf.string),
        })
    hazed_image = tf.decode_raw(img_features['hazed_image_raw'], tf.uint8)
    hazed_image = tf.reshape(hazed_image, [height, width, 3])
    if df.FLAGS.use_fp16:
        hazed_image = tf.image.convert_image_dtype(hazed_image, tf.float16)
    else:
        hazed_image = tf.image.convert_image_dtype(hazed_image, tf.float32)
    clear_image = tf.decode_raw(img_features['clear_image_raw'], tf.uint8)
    clear_image = tf.reshape(clear_image, [height, width, 3])
    if df.FLAGS.use_fp16:
        clear_image = tf.image.convert_image_dtype(clear_image, tf.float16)
    else:
        clear_image = tf.image.convert_image_dtype(clear_image, tf.float32)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    return _generate_image_batch(hazed_image, clear_image, min_queue_examples, batch_size, shuffle=False)


if __name__ == '__main__':
    # Unit test
    file_names = []
    image_list=[]
    clear_dict = {}
    image_input("./ClearImages", file_names, image_list, clear_dict=clear_dict,clear_image=True)
    image_list_shuffle(image_list)
    print(file_names)
    print(image_list)
    print(clear_dict)
    image_list = _read_image(image_list)
    for img in image_list:
        print(tf.size(img.image_tensor))
    a = [1,2,3]
    print(tf.size(a))
    print(np.shape(a))
    # get_image_batch(1, 2, image_list, clear_dict)
    image_tensor = tf.image.decode_jpeg("./ testd / test1 / UNADJUSTEDNONRAW_thumb_286.jpg", channels=dehazenet.RGB_CHANNEL)
    print(tf.size(image_tensor))

