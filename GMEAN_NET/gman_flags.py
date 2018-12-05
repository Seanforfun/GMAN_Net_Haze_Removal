import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
# Dehazenet actually doesn't require very high float data accuracy,
# so fp16 is normally set False
# TODO Need to be discussed
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 35,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('input_image_height', 224,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 224,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('original_height', 100000,
                            """Input image original height.""")
tf.app.flags.DEFINE_integer('original_width', 100000,
                            """Input image original width.""")
tf.app.flags.DEFINE_string('haze_train_images_dir', './HazeImages/TrainImages',
                           """Path to the hazed train images directory.""")
tf.app.flags.DEFINE_string('clear_train_images_dir', './ClearImages/TrainImages',
                           """Path to the clear train images directory.""")
tf.app.flags.DEFINE_string('tfrecord_format', 'gman-%d.tfrecords',
                           """Format of tf-records, file name must end with -index_number.""")
tf.app.flags.DEFINE_string('tfrecord_json', './TFRecord/tfrecords.json',
                           """Json file to save the status of tfrecords.""")
tf.app.flags.DEFINE_string('tfrecord_path', './TFRecord',
                           """Path to save tfrecords.""")
tf.app.flags.DEFINE_boolean('tfrecord_rewrite', False,
                            """Whether to delete and rewrite the TFRecord.""")
tf.app.flags.DEFINE_string('PerceNet_dir', './PerceNetModel/vgg16.npy',
                           """Path to save the PerceNet Model""")
tf.app.flags.DEFINE_boolean('train_restore', True,
                            """Whether to restore the trained model.""")
tf.app.flags.DEFINE_string('train_json_path', './DeHazeNetModel/trainFlowControl.json',
                           """Path to save training status json file.""")
tf.app.flags.DEFINE_integer('max_epoch', 500,
                            """Max epoch number for training.""")
tf.app.flags.DEFINE_string('train_learning_rate', './DeHazeNetModel/trainLearningRate.json',
                           """Path to save training learning rate json file.""")


# Some systematic parameters
tf.app.flags.DEFINE_string('train_dir', './DeHazeNetModel',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 19990000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Variables for evaluation
tf.app.flags.DEFINE_string('eval_dir', './DeHazeNet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './DeHazeNetModel',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('haze_test_images_dir', './HazeImages/TestImages',
                           """Path to the hazed test images directory.""")
tf.app.flags.DEFINE_string('clear_test_images_dir', './ClearImages/TestImages',
                           """Path to the clear train images directory.""")
tf.app.flags.DEFINE_string('clear_result_images_dir', './ClearResultImages/',
                           """Path to the dehazed test images directory.""")
tf.app.flags.DEFINE_string('tfrecord_eval_path', './TFRecord/eval.tfrecords',
                           """Path to save the test TFRecord of the images""")
tf.app.flags.DEFINE_boolean('tfrecord_eval_rewrite', False,
                            """Whether to delete and rewrite the TFRecord.""")
tf.app.flags.DEFINE_string('save_image_type', 'jpg',
                            """In which format to save image.""")

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
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 10,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000000000,
                            """Number of test examples to run""")
tf.app.flags.DEFINE_boolean('eval_only_haze', False,
                            """Whether to load clear images.""")
