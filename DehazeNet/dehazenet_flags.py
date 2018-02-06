import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
# Dehazenet actually doesn't require very high float data accuracy,
# so fp16 is normally set False
# TODO Need to be discussed
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('input_image_height', 128,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 128,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('original_height', 100000,
                            """Input image original height.""")
tf.app.flags.DEFINE_integer('original_width', 100000,
                            """Input image original width.""")
tf.app.flags.DEFINE_string('haze_train_images_dir', './HazeImages/TrainImages',
                           """Path to the hazed train images directory.""")
tf.app.flags.DEFINE_string('clear_train_images_dir', './ClearImages/TrainImages',
                           """Path to the clear train images directory.""")


# Some systematic parameters
tf.app.flags.DEFINE_string('train_dir', './DeHazeNetModel',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Variables for evaluation
tf.app.flags.DEFINE_string('eval_dir', './DeHazeNet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './DeHazeNetEval',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('haze_test_images_dir', './HazeImages/TestImages',
                           """Path to the hazed test images directory.""")
tf.app.flags.DEFINE_string('clear_test_images_dir', './ClearImages/TestImages',
                           """Path to the clear train images directory.""")
tf.app.flags.DEFINE_string('clear_result_images_dir', './ClearResultImages',
                           """Path to the hazed test images directory.""")

tf.app.flags.DEFINE_boolean('eval_log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('eval_max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('eval_num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('eval_input_image_height', 128,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('eval_input_image_width', 128,
                            """Input image width.""")