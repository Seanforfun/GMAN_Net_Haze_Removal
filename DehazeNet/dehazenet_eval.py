#  ====================================================
#   Filename: dehazenet_eval.py
#   Function: This file is used for evaluate our model and create a
#   image from a hazed image.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import dehazenet_input as di
import dehazenet_tools as dt
import dehazenet_eval as de
import dehazenet_multi_gpu_train as dmgt
import dehazenet as dn
import dehazenet_flags as df
import numpy as np
import skimage.io as io
from  skimage import measure
from skimage import transform
from PIL import Image as im
import math
from Image import *
import cv2
import matplotlib.image as mpimg

import re
from datetime import datetime
import os.path
import time

EVAL_TOWER_NAME = "tower"
EVAL_MOVING_AVERAGE_DECAY = 0.9999
# Frames used to save clear training image information
_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}
# Frames used to save hazed training image information
_hazed_test_file_names = []
_hazed_test_img_list = []
IMAGE_JPG_FORMAT = 'jpg'
IMAGE_PNG_FORMAT = 'png'
SINGLE_IMAGE_NUM = 1

IMAGE_INDEX = 0
IMAGE_A = 1
IMAGE_BETA = 2

_hazed_test_A_dict = {}
_hazed_test_beta_dict = {}


def eval_hazed_input(dir, file_names, image_list, dict_A, dict_beta):
    if not dir:
        raise ValueError('Please supply a data_dir')
    file_list = os.listdir(dir)
    for filename in file_list:
        if os.path.isdir(os.path.join(dir, filename)):
            eval_hazed_input(os.path.join(dir, filename), file_names, image_list, dict_A, dict_beta)
            pass
        elif filename.endswith(".png") | filename.endswith(".jpg") | filename.endswith(".bmp"):
            file_names.append(filename)
            temp_name = filename[0:(len(filename) - 4)]
            hazed_image_split = temp_name.split('_')
            file_name = os.path.join(dir, filename)
            current_image = Image(path=file_name)
            current_image.image_index = hazed_image_split[IMAGE_INDEX]
            image_list.append(current_image)
            if hazed_image_split[IMAGE_A] not in dict_A:
                A_list = []
                A_list.append(current_image)
                dict_A[hazed_image_split[IMAGE_A]] = A_list
            else:
                dict_A[hazed_image_split[IMAGE_A]].append(current_image)
            if hazed_image_split[IMAGE_BETA] not in dict_beta:
                beta_list = []
                beta_list.append(current_image)
                dict_beta[hazed_image_split[IMAGE_BETA]] = beta_list
            else:
                dict_beta[hazed_image_split[IMAGE_BETA]].append(current_image)
    return file_names, image_list, dict_A, dict_beta

def lz_net_eval(hazed_batch, height, width):
    x_s = dt.conv_eval('DN_conv1_1', hazed_batch, 3, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv1_2', x_s, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x = dt.conv_eval('upsampling_1', x, 64, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])
    x = dt.conv_eval('upsampling_2', x, 128, 128, kernel_size=[3, 3], stride=[1, 2, 2, 1])

    x1 = dt.conv_eval('DN_conv1_3', x, 128, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv2_1', x1, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti_eval('DN_conv2_2', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x1)
    # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    # x = tools.conv('DN_conv2_3', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv2_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x2 = dt.conv_eval('DN_conv3_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv3_2', x2, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti_eval('DN_conv3_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x2)
    # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    # x = tools.conv('DN_conv3_4', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv3_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x3 = dt.conv_eval('DN_conv4_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv4_2', x3, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv4_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti_eval('DN_conv4_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x3)
    # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    x = dt.conv_eval('DN_conv4_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x4 = dt.conv_eval('DN_conv5_1', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv5_2', x4, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv5_3', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_eval('DN_conv5_4', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti_eval('DN_conv5_5', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = tf.add(x, x4)
    # x = tools.batch_norm(x)
    x = dt.acti_layer(x)

    x = dt.deconv_eval('DN_deconv1', x, 64, 64, output_shape=[1, int((height + 1)/2), int((width + 1)/2), 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])

    x = dt.deconv_eval('DN_deconv2', x, 64, 64, output_shape=[1, height, width, 64], kernel_size=[3, 3], stride=[1, 2, 2, 1])
    x = dt.conv_eval('DN_conv6_6', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    x = dt.conv_nonacti_eval('DN_conv6_7', x, 64, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1])

    x = tf.add(x, x_s)
    x = dt.acti_layer(x)

    x = dt.conv_eval('DN_conv6_8', x, 64, 3, kernel_size=[3, 3], stride=[1, 1, 1, 1])
    return x

def _eval_generate_image_batch(hazed_image, min_queue_examples, batch_size, shuffle=True):
    num_preprocess_threads = 8

    if shuffle:
        h_images, c_images = tf.train.shuffle_batch(
            [hazed_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        h_images, c_images = tf.train.batch(
            [hazed_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # Display the training images in the visualizer.
    tf.summary.image('hazed_images', h_images)
    return h_images, c_images


def tf_psnr(im1, im2):
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def cal_psnr(im1, im2):
    '''
        assert pixel value range is 0-255 and type is uint8
    '''
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def eval_once(saver, train_op, hazed_images, clear_images, hazed_images_obj_list, index, placeholder, psnr_list, ssim_list, heights, widths):
    with  tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=df.FLAGS.log_device_placement)) as sess:
        ckpt = tf.train.get_checkpoint_state(df.FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        temp_image_list = []
        temp_image_list.append(hazed_images[index])
        prediction = sess.run([train_op], feed_dict={placeholder[index]: temp_image_list})
        # Run the session and get the prediction of one clear image
        dehazed_image = write_images_to_file(prediction, hazed_images_obj_list[index], heights[index], widths[index], sess)
        if clear_images is not None:
            clear_image = np.uint8(clear_images[index] * 255)
            psnr_value = cal_psnr(dehazed_image, clear_image)
            ssim_value = measure.compare_ssim(dehazed_image, clear_image, multichannel=True)
            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
            print('-------------------------------------------------------------------------------------------------------------------------------')
            format_str = ('%s: image: %s PSNR: %f; SSIM: %f')
            print(format_str % (datetime.now(), hazed_images_obj_list[index].path, psnr_value, ssim_value))
            print('-------------------------------------------------------------------------------------------------------------------------------')


def evaluate_cartesian_product():
    with tf.Graph().as_default() as g:
        # A list used to save all hazed images
        psnr_list = []
        hazed_image_list = []
        clear_image_list = []
        # Read all hazed images and clear images from directory and save into memory
        di.image_input(df.FLAGS.clear_test_images_dir, _clear_test_file_names, _clear_test_img_list,
                       _clear_test_directory, clear_image=True)
        if len(_clear_test_img_list) == 0:
            raise RuntimeError("No image found! Please supply clear images for training or eval ")
        # Hazed training image pre-process
        eval_hazed_input(df.FLAGS.haze_test_images_dir, _hazed_test_file_names, _hazed_test_img_list, _hazed_test_A_dict, _hazed_test_beta_dict)

        if len(_hazed_test_img_list) == 0:
            raise RuntimeError("No image found! Please supply hazed images for training or eval ")
        haze_image_obj_list = []
        for k, v in _hazed_test_A_dict.items():
            for image in v:
                hazed_image = im.open(image.path)
                haze_image_obj_list.append(image)
                shape = np.shape(hazed_image)
                # left, upper, right, lower
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
                reshape_hazed_image_arr = np.array(reshape_hazed_image)
                float_hazed_image = reshape_hazed_image_arr.astype('float32') / 255
                hazed_image_list.append(float_hazed_image)

                # arr = np.resize(arr, [224, 224])
                clear_image = di.find_corres_clear_image(image, _clear_test_directory)
                shape = np.shape(clear_image)
                # left, upper, right, lower
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
                reshape_clear_image_arr = np.array(reshape_clear_image)
                float_clear_image = reshape_clear_image_arr.astype('float32') / 255
                clear_image_list.append(float_clear_image)
        for k, v in _hazed_test_beta_dict.items():
            for image in v:
                hazed_image = im.open(image.path)
                haze_image_obj_list.append(image)
                shape = np.shape(hazed_image)
                # left, upper, right, lower
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
                reshape_hazed_image_arr = np.array(reshape_hazed_image)
                float_hazed_image = reshape_hazed_image_arr.astype('float32') / 255
                hazed_image_list.append(float_hazed_image)

                # arr = np.resize(arr, [224, 224])
                clear_image = di.find_corres_clear_image(image, _clear_test_directory)
                shape = np.shape(clear_image)
                # left, upper, right, lower
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
                reshape_clear_image_arr = np.array(reshape_clear_image)
                float_clear_image = reshape_clear_image_arr.astype('float32') / 255
                clear_image_list.append(float_clear_image)

        if len(clear_image_list) != len(hazed_image_list):
            raise RuntimeError("hazed images cannot correspond to clear images!")

        hazed_image = tf.placeholder(tf.float32,
                                     shape=[1, df.FLAGS.input_image_height, df.FLAGS.input_image_width,
                                            dn.RGB_CHANNEL])

        # logist = dmgt.inference(hazed_image)
        logist = lz_net_eval(hazed_image,1,1)
        variable_averages = tf.train.ExponentialMovingAverage(
            dn.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # TODO Zheng Liu please remove the comments of next two lines and add comment to upper five lines
        # logist = lz_net(hazed_image)
        # saver = tf.train.Saver()
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(df.FLAGS.eval_dir, g)
        for index in range(len(clear_image_list)):
            eval_once(saver, summary_writer, logist, summary_op, hazed_image_list, clear_image_list, haze_image_obj_list, index, hazed_image, psnr_list)

        sum = 0
        for psnr in psnr_list:
            sum += psnr
        psnr_avg = sum / len(psnr_list)
        print('Average PSNR: ')
        print(psnr_avg)


def evaluate():
    with tf.Graph().as_default() as g:
        # A list used to save all hazed images
        psnr_list = []
        ssim_list = []
        hazed_image_list = []
        clear_image_list = []
        hazed_image_placeholder_list = []
        height_list = []
        width_list = []
        # Read all hazed images and clear images from directory and save into memory
        if not df.FLAGS.eval_only_haze:
            di.image_input(df.FLAGS.clear_test_images_dir, _clear_test_file_names, _clear_test_img_list,
                           _clear_test_directory, clear_image=True)
            if len(_clear_test_img_list) == 0:
                raise RuntimeError("No image found! Please supply clear images for training or eval ")
        # Hazed training image pre-process
        di.image_input(df.FLAGS.haze_test_images_dir, _hazed_test_file_names, _hazed_test_img_list,
                       clear_dict=None, clear_image=False)
        if len(_hazed_test_img_list) == 0:
            raise RuntimeError("No image found! Please supply hazed images for training or eval ")

        for image in _hazed_test_img_list:
            # Read image from files and append them to the list
            # hazed_image = im.open(image.path)
            hazed_image = im.open(image.path)
            hazed_image = hazed_image.convert('RGB')
            shape = np.shape(hazed_image)
            height_list.append(shape[0])
            width_list.append(shape[1])
            hazed_image_placeholder = tf.placeholder(tf.float32,
                                         shape=[1, shape[0], shape[1], dn.RGB_CHANNEL])
            hazed_image_placeholder_list.append(hazed_image_placeholder)
            hazed_image_arr = np.array(hazed_image)
            float_hazed_image = hazed_image_arr.astype('float32') / 255
            hazed_image_list.append(float_hazed_image)
            if not df.FLAGS.eval_only_haze:
                clear_image = di.find_corres_clear_image(image, _clear_test_directory)
                clear_image_arr = np.array(clear_image)
                float_clear_image = clear_image_arr.astype('float32') / 255
                clear_image_list.append(float_clear_image)


        if not df.FLAGS.eval_only_haze:
            if len(clear_image_list) != len(hazed_image_list):
                raise RuntimeError("hazed images cannot correspond to clear images!")
        print("==================================================================")
        print(len(hazed_image_list))
        print("==================================================================")
        for index in range(len(hazed_image_list)):
            # logist = dmgt.inference(hazed_image)
            logist = lz_net_eval(hazed_image_placeholder_list[index], height_list[index], width_list[index])
            variable_averages = tf.train.ExponentialMovingAverage(
                dn.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            if not df.FLAGS.eval_only_haze:
                eval_once(saver, logist, hazed_image_list, clear_image_list, _hazed_test_img_list, index, hazed_image_placeholder_list, psnr_list, ssim_list, height_list, width_list)
            else:
                eval_once(saver, logist, hazed_image_list, None, _hazed_test_img_list, index,
                          hazed_image_placeholder_list, psnr_list, ssim_list, height_list, width_list)

        if not df.FLAGS.eval_only_haze:
            psnr_avg = cal_average(psnr_list)
            format_str = ('%s: Average PSNR: %5f')
            print(format_str % (datetime.now(), psnr_avg))
            ssim_avg = cal_average(ssim_list)
            format_str = ('%s: Average SSIM: %5f')
            print(format_str % (datetime.now(), ssim_avg))


def cal_average(list):
    sum = 0
    for num in list:
        sum += num
    avg = sum / len(list)
    return avg


def write_images_to_file(logist, image, height, width, sess):
    array = np.reshape(logist[0], newshape=[height, width, dn.RGB_CHANNEL])
    array = array * 255
    # arr1 = np.uint8(array)
    array = tf.saturate_cast(array, dtype=tf.uint8)
    arr1 = sess.run(array)
    result_image = im.fromarray(arr1, 'RGB')
    image_name_base = image.image_index
    # cv2.imwrite(df.FLAGS.clear_result_images_dir + image_name_base + "_" + str(time.time()) + '_pred.jpg', arr1)
    result_image.save(df.FLAGS.clear_result_images_dir + image_name_base + "_" + str(time.time()) + '_pred.jpg', 'jpeg')
    return arr1


def main(self):
    if tf.gfile.Exists(df.FLAGS.clear_result_images_dir):
        tf.gfile.DeleteRecursively(df.FLAGS.clear_result_images_dir)
    tf.gfile.MakeDirs(df.FLAGS.clear_result_images_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
