#  ====================================================
#   Filename: gman_config.py
#   Author: Seanforfun
#   Function: File to save configure constants.
#  ====================================================
import tensorflow as tf

import gman_flags as df

# Configure for training
CONFIG_TRAINING_TRAIN_RESTORE = "train_restore"


def config_load_config():
    config = {CONFIG_TRAINING_TRAIN_RESTORE: df.FLAGS.train_restore}
    return config


def config_update_config(config):
    if not config[CONFIG_TRAINING_TRAIN_RESTORE]:
        config[CONFIG_TRAINING_TRAIN_RESTORE] = True


if __name__ == '__main__':
    pass
