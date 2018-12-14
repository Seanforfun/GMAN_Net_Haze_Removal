#  ====================================================
#   Filename: distribute_flags.py
#   Author: Botao Xiao
#   Function: This is file used to save the flags we can call them
#   using command line.
#  ====================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Distributed training options
tf.app.flags.DEFINE_string('project_name', 'Your project name',
                           """String to save the project name.""")
tf.app.flags.DEFINE_string('job_name', '',
                           """One of ps and worker""")
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """The hosts that runs as parameter server.""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """The hosts that works as workers who are responsible for processing.""")
tf.app.flags.DEFINE_integer("task_index", None,
                            "Worker task index, should be >= 0. task_index=0 is "
                            "the master worker task the performs the variable "
                            "initialization ")
tf.app.flags.DEFINE_integer("replicas_to_aggregate", None,
                            "Number of replicas to aggregate before parameter update"
                            "is applied (For sync_replicas mode only; default: "
                            "num_workers)")


tf.app.flags.DEFINE_integer('intra_op_parallelism_threads ', 0,
                            """
                              Number of threads to use for intra-op parallelism. When training on CPU
                              set to 0 to have the system pick the appropriate number or alternatively
                              set it to the number of physical CPU cores.
                            """)
tf.app.flags.DEFINE_integer('inter_op_parallelism_threads ', 0,
                            """
                             Number of threads to use for inter-op parallelism. If set to 0, the
                            system will pick an appropriate number.
                            """)

# Training options
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """
                           Whether to log device placement.
                           """)

# Training parameters
tf.app.flags.DEFINE_integer('input_image_height', 224,
                            """Input image height.""")
tf.app.flags.DEFINE_integer('input_image_width', 224,
                            """Input image width.""")
tf.app.flags.DEFINE_integer('sample_number', 100000,
                            """Total sample numbers to train.""")
tf.app.flags.DEFINE_float('train_learning_rate', 0.001,
                          """Value of initial learning rate.""")

# Files position
tf.app.flags.DEFINE_string('learning_rate_json', 'YOUR LEARNING RATE SAVING PATH',
                           """Path to save learning rate json file.""")


if __name__ == '__main__':
    pass
