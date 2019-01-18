#  ====================================================
#   Filename: gman_model.py
#   Function: This file is used to save the model of the gmean CNN
#   net.
#  ====================================================
import abc
import tensorflow as tf

import gman_tools as tools
import gman_flags as flags


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def inference(self, input_data):
        pass


# This model consists with the one metioned in paper
# Generic Model-Agnostic Convolutional Neural Network for Single Image Dehazing
# https://arxiv.org/ptools/1810.02862.ptools
class GMAN_V1(Model):

    def inference(self, input_data, batch_size=None, h=None, w=None):
        """
        The forward process of network.
        :param input_data:  Batch used to for training, always in size of [batch_size, h, w, 3]
        :param batch_size:  1 for evaluation and custom number for training.
        :param h: height of the image
        :param w: width of the image
        :return: The result processed by gman
        """
        if h is None or w is None or batch_size is None:
            h = flags.FLAGS.input_image_height
            w = flags.FLAGS.input_image_width
            batch_size = flags.FLAGS.batch_size
        with tf.variable_scope('DehazeNet'):
            x_s = input_data
            # ####################################################################
            # #####################Two convolutional layers###########################
            # ####################################################################
            x = tools.conv('DN_conv1_1', x_s, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv1_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            # ####################################################################
            # ###################Two Downsampling layers############################
            # ####################################################################
            x = tools.conv('upsampling_1', x, 64, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
            x = tools.conv('upsampling_2', x, 128, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])

            # ####################################################################
            # #######################Residual Blocks#################################
            # ####################################################################
            x1 = tools.conv('DN_conv2_1', x, 128, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv2_2', x1, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv2_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x1)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv2_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x2 = tools.conv('DN_conv3_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv3_2', x2, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv3_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x2)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x3 = tools.conv('DN_conv4_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_2', x3, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv4_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x3)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x4 = tools.conv('DN_conv5_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_2', x4, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv5_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x4)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # ####################################################################
            # #####################Two deconvolutional layers#########################
            # ####################################################################
            x = tools.deconv('DN_deconv1', x, 64, 64, output_shape=[batch_size, int((h + 1)/2), int((w + 1)/2), 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
            x = tools.deconv('DN_deconv2', x, 64, 64, output_shape=[batch_size, h, w, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])

            x_r = tools.conv_nonacti('DN_conv7_1', x, 64, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x_r = tf.add(x_r, x_s)
            x_r = tools.acti_layer(x_r)

            return x_r


class GMAN(Model):
    def __init__(self):
        pass

    def inference(self, input_data):
        with tf.variable_scope('DehazeNet'):
            # ################################################################
            # ####################Convolutional Model###########################
            # ################################################################
            x_s = input_data
            x = tools.conv('DN_conv1_1', x_s, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv1_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            # with tf.name_scope('pool1'):
            #     x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)
            #
            # with tf.name_scope('pool2'):
            #     x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

            x = tools.conv('upsampling_1', x, 64, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
            x = tools.conv('upsampling_2', x, 128, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])

            x1 = tools.conv('DN_conv2_1', x, 128, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv2_2', x1, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv2_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x1)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv2_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x2 = tools.conv('DN_conv3_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv3_2', x2, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv3_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x2)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x3 = tools.conv('DN_conv4_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_2', x3, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv4_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv4_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x3)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            # x = tools.conv('DN_conv4_5', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

            x4 = tools.conv('DN_conv5_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_2', x4, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv5_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x4)
            # x = tools.batch_norm(x)
            x = tools.acti_layer(x)

            x5 = tools.conv('DN_conv5_6', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv5_7', x5, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv_nonacti('DN_conv5_8', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x5)
            x = tools.acti_layer(x)

            x = tools.deconv('DN_deconv1', x, 64, 64, output_shape=[35, 112, 112, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
            x = tools.deconv('DN_deconv2', x, 64, 64, output_shape=[35, 224, 224, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])

            # x = tools.conv('DN_conv6_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('DN_conv6_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            # x = tf.add(x, x_s)
            # # # x = tools.batch_norm(x)
            # x = tools.acti_layer(x)

            x = tools.conv_nonacti('DN_conv7_1', x, 64, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tf.add(x, x_s)
            x = tools.acti_layer(x)
            return x


# This model is the one that we started our haze removal design
# Using at 01 - 02/2018
class BeginModel(Model):
    def __init__(self):
        pass

    def inference(self, input_data):
        """
        :param input_data: The hazed training images from get_distorted_image.
        Each image is in the form of Images. 4D tensor of [batch_size, height, witoolsh, 3] size
        Please refer to CIFAR-10 CNN model to design our dehazenet.
        :return: A image batch after trained by CNN
        """
        with tf.name_scope('DehazeNet'):
            x = tools.conv('conv1_1', input_data, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv1_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            # with tf.name_scope('pool1'):
            #     x = tools.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

            x = tools.conv('conv2_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv2_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            # with tf.name_scope('pool2'):
            #     x = tools.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

            x = tools.conv('conv3_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv3_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv3_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            # with tf.name_scope('pool3'):
            #     x = tools.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

            x = tools.conv('conv4_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv4_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv4_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            # with tf.name_scope('pool4'):
            #     x = tools.pool('pool4', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

            x = tools.conv('conv5_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv5_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv5_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            # with tf.name_scope('pool5'):
            #     x = tools.pool('pool5', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

            x = tools.conv('conv6_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
            x = tools.conv('conv6_2', x, 64, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])

        return x


if __name__ == '__main__':
    pass
