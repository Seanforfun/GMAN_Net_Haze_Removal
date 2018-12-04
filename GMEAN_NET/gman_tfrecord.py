#  ====================================================
#   Filename: gman_tfrecord.py
#   Author: Seanforfun
#   Function: This file is used to create tf-records for training.
#  ====================================================
import gman_input as di
import gman_flags as df


# Frames used to save clear training image information
_clear_train_file_names = []
_clear_train_img_list = []
_clear_train_directory = {}
# Frames used to save hazed training image information
_hazed_train_file_names = []
_hazed_train_img_list = []

_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}


def tfrecord_create_tf_record(path):
    di.image_input(df.FLAGS.clear_train_images_dir, _clear_train_file_names, _clear_train_img_list,
                   _clear_train_directory, clear_image=True)
    if len(_clear_train_img_list) == 0:
        raise RuntimeError("No image found! Please supply clear images for training or eval ")
    # Hazed training image pre-process
    di.image_input(df.FLAGS.haze_train_images_dir, _hazed_train_file_names, _hazed_train_img_list,
                   clear_dict=None, clear_image=False)
    if len(_hazed_train_img_list) == 0:
        raise RuntimeError("No image found! Please supply hazed images for training or eval ")

    # Write data into a TFRecord saved in path ./TFRecord
    di.convert_to_tfrecord(_hazed_train_img_list, _hazed_train_file_names, _clear_train_directory,
                           df.FLAGS.input_image_height, df.FLAGS.input_image_width, path,
                           _clear_test_img_list)


if __name__ == '__main__':
    tfrecord_create_tf_record("")
