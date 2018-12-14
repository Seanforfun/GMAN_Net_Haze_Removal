#  ====================================================
#   Filename: distribute_input.py
#   Author: Botao Xiao
#   Function: This file contains the input module of the distributed
#   system.
#  ====================================================
import os
import abc
import multiprocessing
from enum import Enum
import queue

import tensorflow as tf

import distribute_flags as flags
import distribute_constants as constants


class InputOptions(Enum):
    TF_RECORD = 0
    PLACEHOLDER = 1


class Dataloader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_train_batch(self, *args, **kwargs):
        """
        Users need to implement this method to get input and ground truth.
        data_dir: Place to load data, can be either a string or a tuple contains multiple paths.
        :param args: (Optional) Additional parameters for training.
        :param kwargs: (Optional) Additional dict for training.
        :return: raw_data batch
        :return: ground_truth batch
        """
        pass

    @abc.abstractmethod
    def load_eval_batch(self, *args, **kwargs):
        """
        Abstract method of loading evaluation batch, user must implement this function and return
        raw data and ground truth from the data paths.
        :param data_dir: Path to load data, can be either a string or a tuple saving multiple paths.
        :param args: (Optional) Additional parameters for evaluation.
        :param kwargs: (Optional) Additional dict for evaluation.
        :return: raw_data batch in list
        :return: ground_truth (Optional)in list, Ground truth batch.
        """
        pass

    def _generate_image_batch(self, example_list, min_queue_examples, num_thread, shuffle=True):
        if shuffle:
            examples = tf.train.shuffle_batch(
                example_list,
                batch_size=self.batch_size,
                num_threads=num_thread,
                capacity=min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=0)
        else:
            examples = tf.train.batch(
                example_list,
                batch_size=self.batch_size,
                num_threads=num_thread.NUMBER_PREPROCESS_THREADS,
                capacity=min_queue_examples + 3 * self.batch_size
            )
        return examples


class TFRecordDataLoader(Dataloader, metaclass=abc.ABCMeta):
    type = "TFRecordDataLoader"

    def load_train_batch(self, train_image_filename_queue, *args, **kwargs):
        if train_image_filename_queue is None:
            raise RuntimeError("Cannot find get the queue from tf-record.")
        raw_data, ground_truth = self.load_batch_from_tfrecord(train_image_filename_queue)
        return raw_data, ground_truth

    def load_eval_batch(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __decode_raw_data(self, raw_features, height, width, *args, **kwargs):
        """
        :param raw_features: raw examples retrieved from tf-record file
        :return: sample list [raw data, grounp truth]
        """
        pass

    def load_batch_from_tfrecord(self, filename_queue, *args, **kwargs):
        height = flags.FLAGS.input_image_height
        width = flags.FLAGS.input_image_width
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        raw_features = tf.parse_single_example(
            serialized_example,
            self.features)
        example_list = self.__decode_raw_data(raw_features, height, width, args, kwargs)
        min_queue_examples = int(self.sample_number *  constants.MIN_FRACTION_OF_EXAMPLE_IN_QUEUE)
        batch_data = self._generate_image_batch(example_list,
                                                min_queue_examples,
                                                multiprocessing.cpu_count() * 2,
                                                shuffle=True)
        return batch_data


class PlaceholderDataLoader(Dataloader, metaclass=abc.ABCMeta):
    type = "PlaceholderDataLoader"

    def load_train_batch(self, *args, **kwargs):
        return self.__create_placeholder(args, kwargs)

    def load_eval_batch(self, *args, **kwargs):
        pass

    def load_queue_for_placeholder(self, *args, **kwargs):
        batch_queue = queue.Queue()
        self.__put_names_dict_into_queue(batch_queue, args, kwargs)
        return batch_queue

    @abc.abstractmethod
    def __create_placeholder(self, *args, **kwargs):
        """
        User must implement this method and return the placeholder
        :param args: (Optional) User's parameter. Save height, width etc.
        :param kwargs: (Optional) User's dict. Save height, width etc.
        :return: return placeholders
        """
        pass

    @abc.abstractmethod
    def __put_names_dict_into_queue(self, queue, *args, **kwargs):
        """
        Users must implement this method so that all datas path are arranged as dictionary
        and save into the queue.
        :param queue: queue to insert the samples.
        :param args: (Optional) For user to pass extension parameters.
        :param kwargs: (Optional) For user to pass extension dict.
        :return:
        """
        pass

    @abc.abstractmethod
    def decode_data_from_path_name(self, paths, *args, **kwargs):
        """
        We get the sample name queue at the very begining, now we get the data
        from the path and return them so program can generate a queue.
        :param paths: paths of the data to read from.
        :param args: (Optional) For user to pass extension parameters.
        :param kwargs: (Optional) For user to pass extension dict.
        :return: data Dict, two items {'raw_data': [], 'ground_truth': []}
        """
        pass

    def load_placeholder_data(self, sample_path_queue, *args, **kwargs):
        raw_batch = []
        ground_truth_batch = []
        for i in range(self.batch_size):
            paths = sample_path_queue.get()
            data = self.decode_data_from_path_name(paths, args, kwargs)
            raw_batch.append(data['raw_data'])
            ground_truth_batch.append(data['ground_truth'])
            sample_path_queue.put(paths)
        return raw_batch, ground_truth_batch


class GmanDataLoader(TFRecordDataLoader):

    def _TFRecordDataLoader__decode_raw_data(self, raw_features, height, width, *args, **kwargs):
        """
        :param raw_features: raw examples retrieved from tf-record file
        :return: sample list [raw data, grounp truth]
        """
        hazed_image = tf.decode_raw(raw_features['hazed_image_raw'], tf.uint8)
        hazed_height = tf.cast(raw_features['hazed_height'], tf.int32)
        hazed_width = tf.cast(raw_features['hazed_width'], tf.int32)
        hazed_image = tf.reshape(hazed_image, [hazed_height, hazed_width, 3])
        clear_image = tf.decode_raw(raw_features['clear_image_raw'], tf.uint8)
        clear_image = tf.reshape(clear_image, [hazed_height, hazed_width, 3])
        # stack the haze and clear images on channel axis
        composed_images = tf.concat([hazed_image, clear_image], axis=2)
        croped_composed_images = tf.random_crop(composed_images,
                                                [height, width, 6])
        hazed_image = croped_composed_images[:, :, :3]
        clear_image = croped_composed_images[:, :, 3:]
        hazed_image = tf.image.convert_image_dtype(hazed_image, tf.float32)
        clear_image = tf.image.convert_image_dtype(clear_image, tf.float32)
        return [hazed_image, clear_image]

