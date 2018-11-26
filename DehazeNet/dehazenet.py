#  ====================================================
#   Filename: dehazenet.py
#   Function: This file is entrance of the dehazenet.
#   Most of the parameters are defined in this file.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import dehazenet_multi_gpu_train as dmgt
import dehazenet_flags as df


def main(self):
    if tf.gfile.Exists(df.FLAGS.train_dir):
        tf.gfile.DeleteRecursively(df.FLAGS.train_dir)
    tf.gfile.MakeDirs(df.FLAGS.train_dir)
    if df.FLAGS.tfrecord_rewrite:
        if tf.gfile.Exists('./TFRecord/train.tfrecords'):
            tf.gfile.Remove('./TFRecord/train.tfrecords')
            print('We delete the old TFRecord and will generate a new one in the program.')
    print('start')
    image_number = len(os.listdir(df.FLAGS.haze_train_images_dir))
    dmgt.train('./TFRecord/train.tfrecords', image_number)
    print('end')


if __name__ == '__main__':
    tf.app.run()