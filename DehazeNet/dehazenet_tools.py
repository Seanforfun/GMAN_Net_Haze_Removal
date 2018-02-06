#  ====================================================
#   Filename: dehazenet_tools.py
#   Author: Zheng Liu
#   Reviser: Seanforfun
#   Function: This file contains the tools used to
#   create CNN.
#   We instantiate all variables using tf.get_variable()
#   instead of tf.Variable() in order to share variables
#   across multiple GPU training runs. If we only ran
#   this model on a single GPU, we could simplify this
#   function by replacing all instances of
#   tf.get_variable() with tf.Variable().
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import dehazenet as dn
import dehazenet_flags as df


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    # TODO Need to determine a variable initialization method, currently we use normal
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if df.FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv(layer_name, x, in_channels, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1]):
    with tf.variable_scope(layer_name):
        w = _variable_with_weight_decay(name='weights',
                                             shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                                             stddev=5e-2, wd=0.0)
        b = _variable_on_cpu(name='biases', shape=[out_channels],
                             initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x


def batch_norm(x):
    '''
        Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


def norm(name, x, lsize=4):

    tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    return x


def FC_layer(layer_name, x, out_nodes):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x


def weight(kernel_shape, is_uniform=True):

    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return w


def bias(bias_shape):

    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b


if __name__ == '__main__':
    a = tf.placeholder(tf.float32, shape=[1, 2, 3, 4]);
    print(a.get_shape()[-1])