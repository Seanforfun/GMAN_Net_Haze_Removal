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

import os

from Image import *
import dehazenet as dehazenet
import dehazenet_flags as df


IMAGE_INDEX_BIT = 4
# TODO Need to change value before real operations
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20
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
                | image_filename.endswith(".jpg"):
            file_name = os.path.join(dir, image_filename)
            file_names.append(file_name)
            current_image = Image(path=file_name)
            current_image.key = id(current_image)
            current_image.image_index = image_filename[0:IMAGE_INDEX_BIT]
            image_list.append(current_image)
            # Put all clear images into dictionary
            if clear_image:
                if len(image_filename) < IMAGE_INDEX_BIT + IMAGE_SUFFIX_MIN_LENGTH:
                    raise RuntimeError("Incorrect image name: " + image_filename)
                clear_dict[current_image.image_index] = current_image
    return file_names, image_list, clear_dict


def _read_image(filenames_queue):
    """
    :param image_list: A image list which saves the image objects
    :param file_names: A file name list(Optional)
    :return: A image list whose image_tensor is filled
    """
    reader = tf.WholeFileReader()
    key, value = reader.read(filenames_queue)
    image = tf.image.decode_png(value, channels=dehazenet.RGB_CHANNEL, dtype=tf.uint8)
    # for image in image_list:
    #     image_content = tf.read_file(image.path)
    #     if image.path.endswith(".png"):
    #         image_tensor = tf.image.decode_png(image_content,
    #                                            channels=dehazenet.RGB_CHANNEL, dtype=tf.uint8)
    #     elif image.path.endswith("jpg"):
    #         image_tensor = tf.image.decode_jpeg(image_content, channels=dehazenet.RGB_CHANNEL)
    #     # TODO Need to be modified
    #     # image_tensor = tf.cast(image_tensor, tf.float32)
    #     # rshape = tf.reshape(tf.reduce_mean(image_tensor, [0, 1]), [1, 1, 3])
    #     # image_tensor = image_tensor / rshape * 128
    #     image.image_tensor = image_tensor
    return image


def _generate_image_batch(hazed_image, clear_image, min_queue_examples, batch_size, shuffle=True):
    num_preprocess_threads = 16

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
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('hazed_images', h_images)
    tf.summary.image('clear_images', c_images)
    return h_images, c_images


def _image_pre_process(image, height, width, train=True):
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


def _find_corres_clear_image(image, clear_dict):
    clear_image_obj = clear_dict[image.image_index]
    if clear_image_obj.path is None:
        raise RuntimeError("Fail to load path from dictionary")
    image_content = tf.read_file(clear_image_obj.path)
    if clear_image_obj.path.endswith(".png"):
        tensor = tf.image.decode_png(image_content, channels=dehazenet.RGB_CHANNEL, dtype=tf.uint8)
    elif clear_image_obj.path.endswith("jpg"):
        tensor = tf.image.decode_jpeg(image_content, channels=dehazenet.RGB_CHANNEL)
    return tensor


def _find_corres_clear_image_filenames(hazed_image_list, clear_dict):
    clear_image_name_list = []
    for image in hazed_image_list:
        clear_image_obj = clear_dict[image.image_index]
        clear_image_name_list.append(clear_image_obj.path)
    return clear_image_name_list


def get_distorted_image(image_batch_list, height, width, dict, Train=True, file_names=None):
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
        filename_queue = tf.train.string_input_producer(file_names)
        hazed_original_image = _read_image(filename_queue)
        reshape_hazed_image = _image_pre_process(hazed_original_image, height, width)
        resize_hazed = tf.image.resize_images(reshape_hazed_image, [height, width])
        clear_image_names = _find_corres_clear_image_filenames(image_batch_list, dict)
        clear_filename_queue = tf.train.string_input_producer(clear_image_names)
        clear_original_image = _read_image(clear_filename_queue)
        reshape_clear_image = _image_pre_process(clear_original_image, height, width, train=False)
        resize_clear = tf.image.resize_images(reshape_clear_image, [height, width])
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        return _generate_image_batch(resize_hazed, resize_clear, min_queue_examples, df.FLAGS.batch_size,
                                     shuffle=Train)
        # index = 0
        # # TODO Must be modified
        # for image in original_image_batch:
        #     if image.image_tensor is None:
        #         raise RuntimeError("Failed to read image: " + image.path)
        #     # Random crop is not suitable for DeHazeNet
        #     # image.image_tensor = tf.random_crop(
        #     #     image.image_tensor, [height, width, dehazenet.RGB_CHANNEL])
        #     index += index
        #     reshape_hazed_image = _image_pre_process(image.image_tensor, height, width)
        #     min_fraction_of_examples_in_queue = 0.4
        #     min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
        #                              min_fraction_of_examples_in_queue)
        #     clear_image = _find_corres_clear_image(image, dict)
        #     reshape_clear_image = _image_pre_process(clear_image, height, width, train=False)
        #     # if index % min_queue_examples == 0:
        #     #     print('Filling queue with %d images before starting to train. '
        #     #           'This will take a few minutes.' % min_queue_examples)
        # return _generate_image_batch(reshape_hazed_image, reshape_clear_image, min_queue_examples, df.FLAGS.batch_size,
        #                              shuffle=Train)
    else:
        raise RuntimeError('Error input of method distorted_image')


def _get_image_batch(epoch_index, batch_index, batch_size, hazed_input_img_list, clear_dict):
    image_input_batch = []
    image_truthground_batch = []
    image_truthground_list = []
    index = (epoch_index - 1) * (batch_index - 1) * batch_size
    image_batch_list = hazed_input_img_list[index:index+batch_size]
    image_input_obj_batch = get_distorted_image(image_batch_list, df.FLAGS.input_image_height,
                                                df.FLAGS.input_image_width)
    for img in image_input_obj_batch:
        image_input_batch.append(img.image_tensor)
    for image in image_input_obj_batch:
            # Find corresponding clear image object and add them into image_truthground_list
        clear_image = clear_dict[image.image_index]
        image_truthground_list.append(clear_image)
    image_truthground_obj_batch = get_distorted_image(image_truthground_list, df.FLAGS.input_image_height,
                                                      df.FLAGS.input_image_width)
    for img in image_truthground_obj_batch:
        image_truthground_batch.append(img.image_tensor)
    return image_input_batch, image_truthground_batch


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

