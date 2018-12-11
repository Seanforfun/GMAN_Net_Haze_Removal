#  ====================================================
#   Filename: gman.py
#   Function: This file is entrance of the dehazenet.
#   Most of the parameters are defined in this file.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import queue
import time
import tensorflow as tf
import threading

import gman_train as dmgt
import gman_flags as df
import gman_input as di
import gman_config as dc
import gman_constant as constant


class TrainProducer(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        # json file doesn't exist, we create a new one
        if not tf.gfile.Exists(df.FLAGS.tfrecord_json):
            tfrecord_status = di.input_create_tfrecord_json()
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
            tfrecord_to_train = "./TFRecord/train.tfrecords"
            dmgt.train(tfrecord_to_train, self.image_number, config)
            di.input_modify_flow_control_json(df.FLAGS.train_json_path ,tfrecord_to_train)
            dc.config_update_config(config)
            time.sleep(constant.ONE_SECOND * 60)


def main(self):
    thread_list = []
    daemon = threading.Thread(name='GMEAN_Daemon', daemon=True)
    thread_list.append(daemon)
    daemon.start()
    image_number = len(os.listdir(df.FLAGS.haze_train_images_dir))
    q = queue.Queue()
    gman_consumer = TrainConsumer(q, image_number)
    gman_consumer.start()
    thread_list.append(gman_consumer)
    for thread in thread_list:
        thread.join()


if __name__ == '__main__':
    tf.app.run()
