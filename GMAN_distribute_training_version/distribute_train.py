#  ====================================================
#   Filename: distribute_train.py
#   Author: Botao Xiao
#   Function: The training file is used to save the training process
#  ====================================================

import multiprocessing
import time

import tensorflow as tf

import distribute_constants as constant
import distribute_flags as flags
import distribute_input as Input
import distribute_log as logger
import distribute_net as net
import distribute_tower as tower
from distribute_loss import Loss
import distribute_annotations as annotations


@annotations.get_advice(pre_fn=print("this is pre_fn"))
class Train(object):
    """
    Parameter from annotation Injection.
    1. self.data_loader
    2. self.input_mode: 'Train' or 'Eval'
    3. self.batch_size: Number of sample in one batch
    4. self.epoch_num: Number of time to operate
    5. self.sample_number: Sample number in one epoch.
    6. self.model_dir: Path to load the model.
    7. self.data_dir: Path to retrieve the data.
    8. self.pre_fn: (Optional) A handler executed at the beginning.
    9. self.post_fn: (Optional) A handler executed at the end.
    10. self.pre_process_fn: (Optional) A handler after getting data and before
    going into the net.
    11. self.post_processs_fn: (Optional)  A handler after getting the results from
    the net and before loss calculation.
    12. self.net: A net contains model, which will do forward processing
    13. self.optimizer
    14. self.loss: We call the loss(raw_data, ground truth) to get the loss value.
    """
    @staticmethod
    def __create_done_queue(num_workers):
        with tf.device("/job:ps/task:0"):
            return tf.FIFOQueue(num_workers, tf.int32, shared_name="done_queue0")

    def train(self,
              pre_fn=None,
              post_fn=None,
              pre_process_fn=None,
              post_process_fn=None,
              *args,
              **kwargs):
        """
        :param pre_fn: (Optional) A handler of pre train process.
        :param post_fn: (Optional) A handler of post train process.
        :param pre_process_fn: (Optional) A handler of process raw data and ground truth before pass them to the net.
        :param post_process_fn: (Optional) A handler of post the direct result from the net(before calculating loss)
        :param args: (Optional) User's additional parameters.
        :param kwargs: (Optional) User's additional dict.
        :return:
        """
        # ####################################################################
        # #####################Parameters Loading###############################
        # ####################################################################
        is_chief = self.task_index == 0
        replicas_to_aggregate = flags.FLAGS.replicas_to_aggregate
        total_step = self.epoch_num * (self.sample_number // self.batch_size)

        if self.job_name == 'ps' or self.gpu_num == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = self.gpu_num
            device_type = 'gpu'
        num_workers = self.cluster.num_tasks("worker")
        kill_ps_queue = Train.__create_done_queue(num_workers)
        # ####################################################################################
        # #################################Parameter Server#####################################
        # ####################################################################################
        if self.job_name == "ps":
            with tf.Session(self.server.target) as sess:
                for i in range(num_workers):
                    sess.run(kill_ps_queue.dequeue())
            return

        # ####################################################################################
        # #################################Worker Service######################################
        # ####################################################################################
        worker_device = "/job:worker/task:%d" % self.task_index
        # TODO Re-think if this method is a good way.
        ps_device = "/job:ps/cpu:0"

        # ####################################################################################
        # #############################Pre Train Function ########################################
        # ####################################################################################
        pre_train_result = None
        if pre_fn is not None:
            pre_train_result = pre_fn(args, kwargs)

        # ####################################################################################
        # #############################Training Function ########################################
        # ####################################################################################
        global_step = tf.get_variable('global_step', [], dtype=tf.int64, initializer=tf.constant_initializer(0),
                                      trainable=False)

        with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device=ps_device,
                                                      cluster=self.cluster)):
            tower_grads = []
            tower_losses = []
            tower_logist = []
            with tf.variable_scope(tf.get_variable_scope()):
                current_net = self.net
                for i in range(num_devices):
                    with tf.device('/%s:%d' % (device_type, i)):
                        with tf.name_scope('%s_%d' % (constant.TOWER_NAME, i)) as scope:
                            if self.input_mode == Input.InputOptions.TF_RECORD:
                                train_image_filename_queue = tf.train.string_input_producer(
                                    [self.data_dir], shuffle=True)
                                raw_data, ground_truth = self.data_loader.load_train_batch(train_image_filename_queue)
                            else:
                                sample_path_queue = self.data_loader.load_queue_for_placeholder(self.data_dir)
                                raw_data, ground_truth = self.data_loader.load_train_batch()
                            if pre_process_fn is not None:
                                    raw_data, ground_truth = pre_process_fn(raw_data, ground_truth, args, kwargs)
                            current_tower = tower.Tower(current_net, scope,
                                                        tower_grads,
                                                        raw_data,
                                                        ground_truth,
                                                        self.loss,
                                                        self.optimizer)
                            summaries, loss, logist = current_tower.process(post_process_fn, pre_train_result)
                            tower_losses.append(loss)
                            tower_logist.append(logist)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = tower.Tower.average_gradients(tower_grads)
            loss = tf.reduce_mean(tower_losses, name='loss')

            if replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = replicas_to_aggregate

            optimizer = tf.train.SyncReplicasOptimizer(
                self.optimizer, use_locking=False,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="sync_replicas")

            # Apply the gradients to adjust the shared variables
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            chief_queue_runner = optimizer.get_chief_queue_runner()
            token_nums = max(replicas_to_aggregate - num_workers, 0)
            sync_init_op = optimizer.get_init_tokens_op(token_nums)
            init_op = tf.global_variables_initializer()
            kill_ps_enqueue_op = kill_ps_queue.enqueue(1)

            supervisor = tf.train.Supervisor(
                is_chief=is_chief,
                init_op=init_op,
                logdir=self.model_dir,
                recovery_wait_secs=1,
                global_step=global_step)

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            if is_chief:
                logger.info("Worker %d: Initializing session..." % self.task_index)
            else:
                logger.info("Worker %d: Waiting for session to be initialized..." % self.task_index)
            sess = supervisor.prepare_or_wait_for_session(self.server.target, config=sess_config)

            logger.info("Worker %d: Session initialization complete." % self.task_index)
            if is_chief:
                supervisor.start_queue_runners(sess, [chief_queue_runner])
                sess.run(sync_init_op)

            while not supervisor.should_stop():
                start = time.time()
                if self.input_mode == Input.InputOptions.TF_RECORD:
                    _, step, loss_value = sess.run([train_op, global_step, loss])
                else:
                    raw_data_batch, ground_truth_batch = self.data_loader.load_placeholder_data(sample_path_queue)
                    # Using placeholder
                    _, step, loss_value = sess.run([train_op, global_step, loss], feed_dict={raw_data: raw_data_batch, ground_truth: ground_truth_batch})
                duration = time.time() - start

                if step % 10 == 0:
                    num_examples_per_step = self.batch_size * self.gpu_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = self.gpu_num
                    format_str = ('step %d, loss = %.8f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    logger.info(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                if step >= total_step:
                    break
            sess.run(kill_ps_enqueue_op)
            logger.info('kill_ps_enqueue_op done....')
        supervisor.stop()

        # ####################################################################################
        # #############################Post Train Function #######################################
        # ####################################################################################
        if post_fn is not None:
            post_fn(args, kwargs)

    def run(self):
        self.train(pre_fn=None if not hasattr(self, "pre_fn") else getattr(self, "pre_fn"),
                   post_fn=None if not hasattr(self, "post_fn") else getattr(self, "post_fn"),
                   pre_process_fn=None if not hasattr(self, "pre_process_fn") else getattr(self, "pre_process_fn"),
                   post_process_fn=None if not hasattr(self, "post_process_fn") else getattr(self, "post_process_fn")
        )


if __name__ == '__main__':
    pass
