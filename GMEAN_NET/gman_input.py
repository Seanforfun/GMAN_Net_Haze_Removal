#  ====================================================
#   Filename: gman_input.py
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
from PIL import Image as im
import json

from Image import *
import gman_flags as df
import gman_constant as constant
import  gman_log as logger


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
                | image_filename.endswith(".jpg") | image_filename.endswith(".bmp") | image_filename.endswith(".jpeg"):
            file_name = os.path.join(dir, image_filename)
            current_image = Image(path=file_name)
            current_image.key = id(current_image)
            current_image.image_index = image_filename[0:constant.IMAGE_INDEX_BIT]
            image_list.append(current_image)
            # Put all clear images into dictionary
            if clear_image:
                if len(image_filename) < constant.IMAGE_INDEX_BIT + constant.IMAGE_SUFFIX_MIN_LENGTH:
                    raise RuntimeError("Incorrect image name: " + image_filename)
                clear_dict[current_image.image_index] = current_image
    if not clear_image:
        image_list_shuffle(image_list)
    for image in image_list:
        file_names.append(image.path)
    return file_names, image_list, clear_dict


def _generate_image_batch(hazed_image, clear_image, min_queue_examples, batch_size, shuffle=True, visualization=False):
    if shuffle:
        h_images, c_images = tf.train.shuffle_batch(
            [hazed_image, clear_image],
            batch_size=batch_size,
            num_threads=constant.NUMBER_PREPROCESS_THREADS,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        h_images, c_images = tf.train.batch(
            [hazed_image, clear_image],
            batch_size=batch_size,
            num_threads=constant.NUMBER_PREPROCESS_THREADS,
            capacity=min_queue_examples + 3 * batch_size
        )

    if visualization:
        tf.summary.image('hazed_images', h_images)
        tf.summary.image('clear_images', c_images)
    return h_images, c_images


def find_corres_clear_image(image, clear_dict):
    clear_image_obj = clear_dict[image.image_index]
    if not tf.gfile.Exists(clear_image_obj.path):
        raise RuntimeError("Fail to load path from dictionary: " + clear_image_obj.path)
    clear_image = im.open(clear_image_obj.path)
    clear_image = clear_image.convert('RGB')
    return clear_image


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_to_tfrecord(hazed_image_list, hazed_image_file_names, dict, height, width, tfrecord_path, test_image_list):
    counter = 0
    test_clear_index_list = []
    for image in test_image_list:
        test_clear_index = image.image_index
        test_clear_index_list.append(test_clear_index)
    if len(hazed_image_list) == 0:
        raise RuntimeError("No example found for training! Please check your training data set!")
    for image in hazed_image_list:
        if not tf.gfile.Exists(image.path):
            raise ValueError('Failed to find image from path: ' + image.path)
    print('Start converting data into tfrecords...')
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    try:
        for image in hazed_image_list:
            try:
                if image.image_index in test_clear_index_list:
                    continue
                hazed_image = im.open(image.path)
                hazed_image = hazed_image.convert("RGB")
                hazed_image_shape = np.shape(hazed_image)
                haze_height = hazed_image_shape[0]
                haze_width = hazed_image_shape[1]
                reshape_hazed_image_arr = np.array(hazed_image)
                hazed_image_raw = reshape_hazed_image_arr.tostring()
                # ################Getting corresponding clear images#########################
                clear_image = find_corres_clear_image(image, dict)
                reshape_clear_image_arr = np.array(clear_image)
                clear_image_raw = reshape_clear_image_arr.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'hazed_image_raw': bytes_feature(hazed_image_raw),
                    'clear_image_raw': bytes_feature(clear_image_raw),
                    'hazed_height': int64_feature(haze_height),
                    'hazed_width': int64_feature(haze_width),
                }))
                writer.write(example.SerializeToString())
                counter += 1
            except IOError as e:
                raise RuntimeError('Could not read:', image.path)
    finally:
        writer.close()
    print('Transform done! Totally transformed ' + str(counter * 2) + ' pairs of examples.' )


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
            'hazed_height': tf.FixedLenFeature([], tf.int64),
            'hazed_width': tf.FixedLenFeature([], tf.int64),
        })
    hazed_image = tf.decode_raw(img_features['hazed_image_raw'], tf.uint8)
    hazed_height = tf.cast(img_features['hazed_height'], tf.int32)
    hazed_width = tf.cast(img_features['hazed_width'], tf.int32)
    hazed_image = tf.reshape(hazed_image, [hazed_height, hazed_width, 3])
    clear_image = tf.decode_raw(img_features['clear_image_raw'], tf.uint8)
    clear_image = tf.reshape(clear_image, [hazed_height, hazed_width, 3])
    # stack the haze and clear images on channel axis
    composed_images = tf.concat([hazed_image, clear_image], axis=2)
    croped_composed_images = tf.random_crop(composed_images, [df.FLAGS.input_image_height, df.FLAGS.input_image_width, 6])
    hazed_image = croped_composed_images[:, :, :3]
    clear_image = croped_composed_images[:, :, 3:]
    if df.FLAGS.use_fp16:
        hazed_image = tf.image.convert_image_dtype(hazed_image, tf.float16)
        clear_image = tf.image.convert_image_dtype(clear_image, tf.float16)
    else:
        hazed_image = tf.image.convert_image_dtype(hazed_image, tf.float32)
        clear_image = tf.image.convert_image_dtype(clear_image, tf.float32)
    min_queue_examples = int(constant.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             constant.MIN_FRACTION_OF_EXAMPLE_IN_QUEUE)
    return _generate_image_batch(hazed_image, clear_image, min_queue_examples, batch_size, shuffle=False)


def input_get_queue_from_tfrecord(tfrecords_filename, batch_size, height, width):
    raw_data = read_tfrecords_and_add_2_queue(tfrecords_filename, batch_size, height, width)
    return tf.contrib.slim.prefetch_queue.prefetch_queue(list(raw_data), capacity=2 * df.FLAGS.num_gpus)


# ####################################################################################
# #################################Json file operations for tf-records########################
# ####################################################################################
def input_create_tfrecord_json():
    tfrecord_list = os.listdir(df.FLAGS.tfrecord_path)
    tfrecord_status_file = open(df.FLAGS.tfrecord_json, "w")
    try:
        # create dictionary for tf-record names
        # key(String): name of tf-record : value(Boolean): if existing
        tfrecord_existing_dict = {"tfrecord_status": {}}
        # {filename-0.tfrecords : False ...  filename-max_epoch-1.tfrecords : False}
        for index in range(500):
            tfrecord_name = df.FLAGS.tfrecord_format % index
            if tfrecord_list.__contains__(tfrecord_name):
                tfrecord_existing_dict["tfrecord_status"][tfrecord_name] = constant.INPUT_TFRECORD_COMPLETE
            else:
                tfrecord_existing_dict["tfrecord_status"][tfrecord_name] = constant.INPUT_TFRECORD_NOT_COMPLETE
        json.dump(tfrecord_existing_dict, tfrecord_status_file)
        logger.info("Create Json file for record tf-record.")
    except IOError as err:
        raise RuntimeError("[Error]: Error happens when read/write " + df.FLAGS.tfrecord_json + ".")
    finally:
        tfrecord_status_file.close()
    return tfrecord_existing_dict


def input_load_existing_tfrecords():
    if not os.path.exists(df.FLAGS.tfrecord_json):
        tfrecord_existing_dict = input_create_tfrecord_json()
    else:
        # File exist, we need to load the json object
        tfrecord_status_file = open(df.FLAGS.tfrecord_json, "r")
        try:
            tfrecord_existing_dict = json.load(tfrecord_status_file)
        except IOError as err:
            raise RuntimeError("[Error]: Error happens when read/write " + df.FLAGS.tfrecord_json + ".")
        finally:
            tfrecord_status_file.close()
    return tfrecord_existing_dict["tfrecord_status"]


# ####################################################################################
# #######################Json file operations for model training control########################
# ####################################################################################
def input_create_flow_control_json():
    # Current json file doesn't exist
    flow_control_file = open(df.FLAGS.train_json_path, "w")
    try:
        flow_control = {'train_flow_control': []}
        json.dump(flow_control, flow_control_file)
        logger.info("Create Json file for training flow control.")
    except IOError as err:
        raise RuntimeError("[Error]: Error happens when read/write " + df.FLAGS.train_json_path + ".")
    finally:
        flow_control_file.close()
    return flow_control["train_flow_control"]


def input_load_flow_control_json():
    if not os.path.exists(df.FLAGS.train_json_path):
        flow_control = input_create_flow_control_json()
    else:
        # File exist, we need to load the json object
        flow_control_json_file = open(df.FLAGS.train_json_path, "r")
        try:
            flow_control_map = json.load(flow_control_json_file)
            flow_control = flow_control_map["train_flow_control"]
        except IOError as err:
            raise RuntimeError("[Error]: Error happens when read/write " + df.FLAGS.train_json_path + ".")
        finally:
            flow_control_json_file.close()
    return flow_control


def input_modify_flow_control_json(train_flow_control_json, finished_tfrecord):
    train_flow_control_json_file = open(train_flow_control_json, "r")
    try:
        json_data = json.load(train_flow_control_json_file)
    except IOError:
        raise RuntimeError("[Error]: Error happens when read/write " + train_flow_control_json + ".")
    finally:
        train_flow_control_json_file.close()
    previous_trained = json_data["train_flow_control"]
    assert not previous_trained.__contains__(finished_tfrecord), "[Error]: Trained tf-record was trained again!"
    previous_trained.append(finished_tfrecord)
    # Save updated data into json file
    train_flow_control_json_file = open(train_flow_control_json, "w")
    try:
        json.dump(json_data, train_flow_control_json_file)
    except IOError:
        raise RuntimeError("[Error]: Error happens when read/write " + train_flow_control_json + ".")
    finally:
        train_flow_control_json_file.close()


# ####################################################################################
# ##########################Parse tf-record for traininig####################################
# ####################################################################################
def input_parse_tfrecord_index(name):
    name, _ = os.path.splitext(name)
    return int(name.split('-')[constant.INPUT_TFRECORD_INDEX_POSITION])


def input_parse_tfrecord_for_training(tfrecord_map, trained_model_list, queue):
    trained_model_number = len(trained_model_list)
    if trained_model_number == 0:
        next_index = 0
    else:
        last_finished = trained_model_list[trained_model_number - 1]
        next_index = input_parse_tfrecord_index(last_finished) + 1
    for index in range(next_index, df.FLAGS.max_epoch):
        if tfrecord_map[df.FLAGS.tfrecord_format % index] == constant.INPUT_TFRECORD_COMPLETE:
            queue.put(os.path.join(df.FLAGS.tfrecord_path, df.FLAGS.tfrecord_format % index))


if __name__ == '__main__':
    pass

