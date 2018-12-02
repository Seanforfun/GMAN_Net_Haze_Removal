#  ====================================================
#   Filename: dehazenet.py
#   Function: This file is entrance of the dehazenet.
#   Most of the parameters are defined in this file.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import queue

import tensorflow as tf
import dehazenet_multi_gpu_train as dmgt
import dehazenet_flags as df
import dehazenet_input as di
import dehazenet_config as dc


class TrainProducer(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        # json file doesn't exist, we create a new one
        if not tf.gfile.Exists(df.FLAGS.tfrecord_json):
            di.input_create_tfrecord_json()
            di.input_create_flow_control_json()
        else:
            tfrecord_status = di.input_load_existing_tfrecords()
            flow_control = di.input_load_flow_control_json()
            di.input_parse_tfrecord_for_training(tfrecord_status, flow_control, self.queue)


class TrainConsumer(threading.Thread):
    def __init__(self, queue, image_number):
        super().__init__()
        self.queue = queue
        self.image_number = image_number

    def run(self):
        config = dc.config_load_config()
        while True:
            tfrecord_to_train = self.queue.get()
            dmgt.train(tfrecord_to_train, self.image_number, config)
            di.input_modify_flow_control_json(df.FLAGS.train_json_path ,tfrecord_to_train)
            dc.config_update_config(config)


def main(self):
    if tf.gfile.Exists(df.FLAGS.train_dir):
        tf.gfile.DeleteRecursively(df.FLAGS.train_dir)
    tf.gfile.MakeDirs(df.FLAGS.train_dir)
    if df.FLAGS.tfrecord_rewrite:
        if tf.gfile.Exists('./TFRecord/train.tfrecords'):
            tf.gfile.Remove('./TFRecord/train.tfrecords')
            print('We delete the old TFRecord and will generate a new one in the program.')
    image_number = len(os.listdir(df.FLAGS.haze_train_images_dir))
    q = queue.Queue()
    dmgt.train('./TFRecord/train.tfrecords', image_number)


if __name__ == '__main__':
    tf.app.run()