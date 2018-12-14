#  ====================================================
#   Filename: distribute.py
#   Author: Botao Xiao
#   Function: This is the entrance of the distributed training system.
#   We run the training program by calling this file.
#  ====================================================
import os
import sys

import tensorflow as tf

# ############################################################################################
# ################All modules are reserved for reflection, please don't modify imports####################
# ############################################################################################
import distribute_flags as flags
import distribute_train as Train
import distribute_annotations as annotations
import distribute_model as model
import distribute_input as Input
import distribute_eval as Eval
import distribute_net as net
import distribute_loss as Loss


@annotations.current_model(model='Gman')
@annotations.optimizer(optimizer=tf.train.AdamOptimizer(0.001))
@annotations.loss(loss="GmanLoss")
@annotations.current_mode(mode='Train')
@annotations.current_input(input='GmanDataLoader')
@annotations.current_feature(features={
            'hazed_image_raw': tf.FixedLenFeature([], tf.string),
            'clear_image_raw': tf.FixedLenFeature([], tf.string),
            'hazed_height': tf.FixedLenFeature([], tf.int64),
            'hazed_width': tf.FixedLenFeature([], tf.int64),
        })
@annotations.gpu_num(gpu_num=4)
@annotations.ps_hosts(ps_hosts="127.0.0.1: 8080")
@annotations.worker_hosts(worker_hosts="127.0.0.1:8081")
@annotations.job_name(job_name=flags.FLAGS.job_name)
@annotations.task_index(task_index=flags.FLAGS.task_index)
@annotations.batch_size(batch_size=35)
@annotations.sample_number(sample_number=13990)
@annotations.epoch_num(epoch_num=20)
@annotations.model_dir(model_dir="/home/xiaob6/distribute/Model")
@annotations.data_dir(data_dir="/home/xiaob6/dehaze/dehazenet/TFRecord/train.tfrecords")
def main(self):
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # ######################################################################
    # #######################Work on the annotations###########################
    # ######################################################################
    # Step 1: Get distributed information
    ps_hosts = annotations.get_value_from_annotation(main, "ps_hosts")
    worker_hosts = annotations.get_value_from_annotation(main, "worker_hosts")
    ps_spec = ps_hosts.split(",")
    worker_spec = worker_hosts.split(",")
    job_name = annotations.get_value_from_annotation(main, "job_name")
    task_index = annotations.get_value_from_annotation(main, "task_index")
    if job_name == 'ps':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Step 2: parameters to build the operator
    optimizer = annotations.get_value_from_annotation(main, 'optimizer')
    mode = annotations.get_value_from_annotation(main, 'mode')
    if mode != 'Train' and mode != 'Eval':
        raise ValueError("mode must be set in the annotation @current_mode")
    batch_size = annotations.get_value_from_annotation(main, 'batch_size')
    epoch_num = annotations.get_value_from_annotation(main, 'epoch_num')
    sample_number = annotations.get_value_from_annotation(main, 'sample_number')
    data_dir = annotations.get_value_from_annotation(main, 'data_dir')
    model_dir = annotations.get_value_from_annotation(main, "model_dir")
    if not os.path.exists(model_dir):
        raise ValueError("Path to save or restore model doesn't exist")
    loss = annotations.get_instance_from_annotation(main, 'loss', Loss)
    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})
    cluster.num_tasks("worker")
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    gpu_num = annotations.get_value_from_annotation(main, 'gpu_num')

    # Step 3: Create data loader instance
    data_loader = annotations.get_instance_from_annotation(main, 'input', Input)
    # Step 3.1: Build the data loader instance
    if data_loader.type == "TFRecordDataLoader":
        if not hasattr(main, 'features'):
            raise ValueError("Please use @current_feature to create your features for data_loader")
        features = annotations.get_value_from_annotation(main, 'features')
        setattr(data_loader, 'features', features)
        input_mode = Input.InputOptions.TF_RECORD
    else:
        input_mode = Input.InputOptions.PLACEHOLDER
    setattr(data_loader, 'batch_size', batch_size)
    setattr(data_loader, 'sample_number', sample_number)
    setattr(data_loader, 'data_dir', data_dir)
    setattr(data_loader, 'gpu_num', gpu_num)

    # Step 4: Get operator instance
    mod = sys.modules['__main__']
    operator_module = getattr(mod, mode)
    class_obj = getattr(operator_module, mode, mod)
    operator = class_obj.__new__(class_obj)

    # Step 5: Build the operator
    setattr(operator, 'task_index', task_index)
    setattr(operator, 'job_name', job_name)
    setattr(operator, 'optimizer', optimizer if optimizer is not None else tf.train.AdamOptimizer(0.001))
    setattr(operator, 'server', server)
    setattr(operator, 'cluster', cluster)
    setattr(operator, 'data_loader', data_loader)
    setattr(operator, 'input_mode', input_mode)
    setattr(operator, 'batch_size', batch_size)
    setattr(operator, 'epoch_num', epoch_num)
    setattr(operator, 'sample_number', sample_number)
    setattr(operator, 'model_dir', model_dir)
    setattr(operator, 'data_dir', data_dir)
    if job_name != 'ps':
        experiment_model = annotations.get_instance_from_annotation(main, 'model', model)
        experiment_net = net.Net(model=experiment_model)
        setattr(operator, 'net', experiment_net)
    setattr(operator, 'loss', loss)
    setattr(operator, 'gpu_num', gpu_num)

    # ################################################################################
    # #############################Start the operation####################################
    # ################################################################################
    operator.run()


if __name__ == '__main__':
    tf.app.run()
