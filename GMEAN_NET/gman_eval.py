#  ====================================================
#   Filename: gman_eval.py
#   Function: This file is used for evaluate our model and create a
#   image from a hazed image.
#  ====================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image as im
from skimage import measure

import gman_constant as constant
import gman_flags as df
import gman_input as di
import gman_log as logger
import gman_model as model
from Image import *

# Frames used to save clear training image information
_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}
# Frames used to save hazed training image information
_hazed_test_file_names = []
_hazed_test_img_list = []


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
            current_image.image_index = hazed_image_split[constant.IMAGE_INDEX]
            image_list.append(current_image)
            if hazed_image_split[constant.IMAGE_A] not in dict_A:
                A_list = []
                A_list.append(current_image)
                dict_A[hazed_image_split[constant.IMAGE_A]] = A_list
            else:
                dict_A[hazed_image_split[constant.IMAGE_A]].append(current_image)
            if hazed_image_split[constant.IMAGE_BETA] not in dict_beta:
                beta_list = []
                beta_list.append(current_image)
                dict_beta[hazed_image_split[constant.IMAGE_BETA]] = beta_list
            else:
                dict_beta[hazed_image_split[constant.IMAGE_BETA]].append(current_image)
    return file_names, image_list, dict_A, dict_beta


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


def eval_once(graph, saver, train_op, hazed_image, clear_image, hazed_images_obj, placeholder, psnr_list, ssim_list, h, w):
    with tf.Session(graph= graph, config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=df.FLAGS.log_device_placement,
            gpu_options=tf.GPUOptions(allow_growth=True,
                                              per_process_gpu_memory_fraction=1,
                                              visible_device_list="0"))) as sess:
        ckpt = tf.train.get_checkpoint_state(df.FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        start = time.time()
        prediction = sess.run([train_op], feed_dict={placeholder: [hazed_image]})
        duration = time.time() - start
        # Run the session and get the prediction of one clear image
        dehazed_image = write_images_to_file(prediction, hazed_images_obj, h, w, sess)
        if not df.FLAGS.eval_only_haze:
            psnr_value = cal_psnr(dehazed_image, np.uint(clear_image))
            ssim_value = measure.compare_ssim(np.uint8(dehazed_image), np.uint8(clear_image), multichannel=True)
            ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
            logger.info('-------------------------------------------------------------------------------------------------------------------------------')
            format_str = 'image: %s PSNR: %f; SSIM: %f; (%.4f seconds)'
            logger.info(format_str % (hazed_images_obj.path, psnr_value, ssim_value, duration))
            logger.info('-------------------------------------------------------------------------------------------------------------------------------')
        else:
            print('-------------------------------------------------------------------------------------------------------------------------------')
            format_str = 'image: %s (%.4f seconds)'
            logger.info(format_str % (hazed_images_obj.path, duration))
            print('-------------------------------------------------------------------------------------------------------------------------------')
    sess.close()

def evaluate():
    # A list used to save all psnr and ssim.
    psnr_list = []
    ssim_list = []
    # Read all hazed images indexes and clear images from directory
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
        graph = tf.Graph()
        with graph.as_default():
            # ########################################################################
            # ########################Load images from disk##############################
            # ########################################################################
            # Read image from files and append them to the list
            hazed_image = im.open(image.path)
            hazed_image = hazed_image.convert('RGB')
            shape = np.shape(hazed_image)
            hazed_image_placeholder = tf.placeholder(tf.float32, shape=[constant.SINGLE_IMAGE_NUMBER, shape[0], shape[1], constant.RGB_CHANNEL])
            hazed_image_arr = np.array(hazed_image)
            float_hazed_image = hazed_image_arr.astype('float32') / 255
            if not df.FLAGS.eval_only_haze:
                clear_image = di.find_corres_clear_image(image, _clear_test_directory)
                clear_image_arr = np.array(clear_image)

            # ########################################################################
            # ###################Restore model and do evaluations#####################
            # ########################################################################
            gman = model.GMAN_V1()
            logist = gman.inference(hazed_image_placeholder, batch_size=1, h=shape[0], w=shape[1])
            variable_averages = tf.train.ExponentialMovingAverage(
                constant.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            # saver, train_op, hazed_image, clear_image_arr, hazed_images_obj, placeholder, psnr_list, ssim_list, h, w
            if not df.FLAGS.eval_only_haze:
                eval_once(graph, saver, logist, float_hazed_image, clear_image_arr, image, hazed_image_placeholder,
                          psnr_list, ssim_list, shape[0], shape[1])
            else:
                eval_once(graph, saver, logist, float_hazed_image, None, image, hazed_image_placeholder,
                          psnr_list, ssim_list, shape[0], shape[1])

    if not df.FLAGS.eval_only_haze:
        psnr_avg = cal_average(psnr_list)
        format_str = 'Average PSNR: %5f'
        logger.info(format_str % psnr_avg)
        ssim_avg = cal_average(ssim_list)
        format_str = 'Average SSIM: %5f'
        logger.info(format_str % ssim_avg)


def cal_average(result_list):
    sum_psnr = sum(result_list)
    return sum_psnr / len(result_list)


def write_images_to_file(logist, image, height, width, sess):
    array = np.reshape(logist[0], newshape=[height, width, constant.RGB_CHANNEL])
    array *= 255
    result_image = tf.saturate_cast(array, tf.uint8)
    arr1 = sess.run(result_image)
    result_image = im.fromarray(arr1, 'RGB')
    image_name_base = image.image_index
    result_image.save(df.FLAGS.clear_result_images_dir + image_name_base + "_" + str(time.time()) + '_pred.png')
    return array


def main(self):
    if tf.gfile.Exists(df.FLAGS.clear_result_images_dir):
        tf.gfile.DeleteRecursively(df.FLAGS.clear_result_images_dir)
    tf.gfile.MakeDirs(df.FLAGS.clear_result_images_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
