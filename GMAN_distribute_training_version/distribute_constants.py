#  ====================================================
#   Filename: distribute_constant.py
#   Function: This file is used to define all constant values for all
#   python files.
#  ====================================================

import tensorflow as tf

import distribute_flags as flags

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# Training
INITIAL_LEARNING_RATE = flags.FLAGS.train_learning_rate

# Input
MIN_FRACTION_OF_EXAMPLE_IN_QUEUE = 0.05


if __name__ == '__main__':
    pass
