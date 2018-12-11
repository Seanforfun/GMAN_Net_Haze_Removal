#  ====================================================
#   Filename: gman_constant.py
#   Function: This file is used to define all constant values for all
#   python files.
#  ====================================================

# gman.py
RGB_CHANNEL = 3

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 4     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.01  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# gman_input.py
# index for finding corresponding clear or haze image.
IMAGE_INDEX_BIT = 4
# TODO Need to change value before real operations in order to avoid memory exceeds
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 192780 * 2
IMAGE_SUFFIX_MIN_LENGTH = 4
# Load what percentage of example queue to memory
MIN_FRACTION_OF_EXAMPLE_IN_QUEUE = 0.05
# Number of threads used in pre-process
NUMBER_PREPROCESS_THREADS = 32
TRAIN_STOP_GRACE_PERIOD = 60

INPUT_TFRECORD_NOT_COMPLETE = "INCOMPLETE"
INPUT_TFRECORD_COMPLETE = "COMPLETE"
INPUT_TFRECORD_REMOVED = "REMOVED"

INPUT_TFRECORD_INDEX_POSITION = 1

ONE_SECOND = 1

# gman_train.py
PROGRAM_START = "Program start!"
PROGRAM_END = "Program stops!"
TRAIN_GPU_MEMORY_ALLOW_GROWTH = False
TRAIN_GPU_MEMORY_FRACTION = 1
TRAIN_VISIBLE_GPU_LIST = "0, 1, 2, 3"

# gman_eval.py
# Decode the name of the haze images.
IMAGE_INDEX = 0
IMAGE_A = 1
IMAGE_BETA = 2

SINGLE_IMAGE_NUMBER = 1

# gman_transmission.py
TRANSMISSION_IMAGE_INDEX = IMAGE_INDEX
TRANSMISSION_IMAGE_A = IMAGE_A
TRANSMISSION_IMAGE_BETA = IMAGE_BETA
TRANSMISSION_CLEAR_DIR = './ClearImages/TestImages'
TRANSMISSION_HAZY_DIR = './HazeImages/TestImages'
TRANSMISSION_TRANSMISSION_DIR = './ClearImages/TransImages'

# gman_options.py
OPTIONS_TRANS_DIR = "./ClearImages/TransImages"
OPTIONS_HAZY_DIR = "./HazeImages/TestImages"
OPTIONS_STATISTICAL_DIR = "./StatisticalFigure"
OPTIONS_THRESHOLD = 0.005
OPTIONS_LOWER_BOUNDARY = 0.7
OPTIONS_STEP_SIZE = 0.01
OPTIONS_TRANSMISSION_THRESHOLD = 0.001
OPTIONS_PNG_SUFFIX = '.png'
OPTIONS_JPG_SUFFIX = '.jpg'
OPTIONS_IMAGE_SUFFIX = OPTIONS_JPG_SUFFIX

# gman_statistic.py
STATS_GROUP_NUM = 10
STATS_CHANNEL_NUM = 3
STATS_CLEAR_INDEX_BIT = 4
STATS_TRANS_INDEX_BIT = 4
STATS_NEED_SERIALIZATION = True
STATS_CLEAR_DICTIONARY = {}
STATS_TRANSMISSION_DICTIONARY = {}
STATS_SERIALIZATION_FILE_NAME = './PQ.pkl'
STATS_START_CALCULATION = True
STATS_SERIALIZATION_BAG = {}

STATS_CLEAR_DIR = "./ClearImages/TestImages"
STATS_RESULT_DIR = "./ClearResultImages"
STATS_TRANSMISSION_DIR = "./ClearImages/TransImages"


if __name__ == '__main__':
    pass
