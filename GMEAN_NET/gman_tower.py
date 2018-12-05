#  ====================================================
#   Filename: gman_tower.py
#   Function: This file is used to save the tower model for multi-gpu
#   training.
#   1. It will deal with the net forward process
#   2. It will calculate the loss for multi-gpu tower
#   3. It will calculate the average gradient and update all models
#  ====================================================
import re
import abc
import tensorflow as tf

import gman_constant as constant


class Tower(metaclass=abc.ABCMeta):
    def __init__(self, net, data_queue, scope, tower_grades, optimizer):
        self.net = net
        self.queue = data_queue
        self.scope = scope
        self.tower_grades = tower_grades
        self.optimizer = optimizer

    @abc.abstractmethod
    def tower_loss(self):
        pass

    @abc.abstractmethod
    def get_gradient(self, loss):
        pass

    @abc.abstractmethod
    def process(self):
        pass

    @staticmethod
    def average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

         Note that this function provides a synchronization point across all towers.

         Args:
           tower_grads: List of lists of (gradient, variable) tuples. The outer list
             is over individual gradients. The inner list is over the gradient
             calculation for each tower.
         Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
         """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


class GMEAN_Tower(Tower):
    @staticmethod
    def __loss(result_batch, ground_truth):
        """
        :param result_batch: A batch of image that been processed by out CNN
        :param ground_truth: The ground truth image to compare with result_batch
        :return: The loss value will be added to tensorflow graph, return is actually not necessary
        but is left here to show respect to CIFAR-10 source code
        """

        # output_per_1, output_per_2, output_per_3 = vgg_per.build(result_batch)
        # output_tru_1, output_tru_2, output_tru_3 = vgg_per.build(clear_image_batch)
        # vgg_tru = Vgg16()
        # vgg_tru.build(clear_image_batch)

        # output_per_1 = vgg_per.conv3_3
        # output_tru_1 = vgg_tru.conv3_3
        #
        # output_per_2 = vgg_per.conv1_1
        # output_tru_2 = vgg_tru.conv1_1
        #
        # output_per_3 = vgg_per.conv2_2
        # output_tru_3 = vgg_tru.conv2_2

        # per_loss = (tf.reduce_mean(tf.square(tf.subtract(output_per_1, output_tru_1))) / 3136) + \
        #            (tf.reduce_mean(tf.square(tf.subtract(output_per_2, output_tru_2))) / 50176) + \
        #            (tf.reduce_mean(tf.square(tf.subtract(output_per_3, output_tru_3))) / 12544)
        loss = tf.reduce_mean(tf.square(tf.subtract(result_batch, ground_truth)))  # + 0.01 * per_loss
        tf.add_to_collection('losses', loss)

        # The total loss is defined as the ms loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def __tower_loss(self, raw_data, ground_truth):
        """Calculate the total loss on a single tower running the DeHazeNet model.

              Args:
                scope: unique prefix string identifying the DEHAZENET tower, e.g. 'tower_0'
                images: Images. 3D tensor of shape [height, width, 3].

              Returns:
                 Tensor of shape [] containing the total loss for a batch of data
              """
        # Put our hazed images into designed CNN and get a result image batch
        logist = self.net.process(raw_data)
        # logist = inference(hazed_batch)
        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        _ = GMEAN_Tower.__loss(logist, ground_truth)
        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', self.scope)
        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses, name='total_loss')

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % constant.TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)
        return total_loss, logist

    def tower_loss(self):
        # Dequeues one batch for the GPU
        hazed_image_batch, clear_image_batch = self.queue.dequeue()
        return GMEAN_Tower.__tower_loss(self, hazed_image_batch, clear_image_batch)

    def get_gradient(self, loss):
        return self.optimizer.compute_gradients(loss)

    def process(self):
        # Calculate the loss for one tower of the dehazenet model. This function
        # constructs the entire dehazenet model but shares the variables across
        # all towers.
        loss, _ = self.tower_loss()

        # Reuse variables for the next tower.
        tf.get_variable_scope().reuse_variables()

        # Retain the summaries from the final tower.
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, self.scope)

        # Calculate the gradients for the batch of data on this GMAN tower.
        grads = self.get_gradient(loss)

        # Keep track of the gradients across all towers.
        self.tower_grades.append(grads)
        return summaries, loss


if __name__ == '__main__':
    pass
