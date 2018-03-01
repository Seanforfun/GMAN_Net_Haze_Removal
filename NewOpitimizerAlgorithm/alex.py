from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import MOptimizer as mo

start_time = time.time()
mnist = input_data.read_data_sets("/home/frr/MNIST_data", one_hot=True)


learning_rate = 0.01
training_iters = 200000
batch_size = 64
display_step = 20

n_input = 784
n_classes = 10
dropout = 0.8

x = tf.placeholder(tf.float32, [None, n_input], name='ImageIn')# Using None is crucial. See Notes
y = tf.placeholder(tf.float32, [None, n_classes], name='LabelIn')# Using None is crucial. See Notes
keep_prob = tf.placeholder(tf.float32, name='KeepProb')

model_path = './simple_mnist.ckpt'# Storing the CNN model after training

def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def alex_net(_X, _weights, _biases, _dropout):

    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])

    pool1 = max_pool('pool1', conv1, k=2)

    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)


    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])

    pool2 = max_pool('pool2', conv2, k=2)

    norm2 = norm('norm2', pool2, lsize=4)

    norm2 = tf.nn.dropout(norm2, _dropout)


    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])

    pool3 = max_pool('pool3', conv3, k=2)

    norm3 = norm('norm3', pool3, lsize=4)

    norm3 = tf.nn.dropout(norm3, _dropout)


    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out


weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64]), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128]), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256]), name = 'wc3'),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024]), name = 'wd1'),
    'wd2': tf.Variable(tf.random_normal([1024, 1024]), name = 'wd2'),
    'out': tf.Variable(tf.random_normal([1024, 10]), name = 'out_w')
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64]), name = 'bc1'),
    'bc2': tf.Variable(tf.random_normal([128]), name = 'bc2'),
    'bc3': tf.Variable(tf.random_normal([256]), name = 'bc3'),
    'bd1': tf.Variable(tf.random_normal([1024]), name = 'bd1'),
    'bd2': tf.Variable(tf.random_normal([1024]), name = 'bd2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name = 'out_b')
}


pred = alex_net(x, weights, biases, keep_prob)

g = tf.get_default_graph()
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name='loss' )
optimizer = mo.MOptimizer(learning_rate=learning_rate, name="testOpt").minimize(cost)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='AdamOpt').minimize(cost)
print('--------------------------------------------')
wc1 = g.get_tensor_by_name('wc1:0')
print(wc1)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'NetAccuracy')

saver = tf.train.Saver()
init = tf.global_variables_initializer()
g = tf.get_default_graph()
# print(g.get_collection('trainable_variables'))
with tf.Session() as sess:

    sess.run(init)

    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
#
    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
    save_path = saver.save(sess, model_path)
#
#     # Storing Parameters in NumPy (.npz) file:
#     wc1 = sess.run(weights['wc1'])
#     wc2 = sess.run(weights['wc2'])
#     wc3 = sess.run(weights['wc3'])
#     wd1 = sess.run(weights['wd1'])
#     wd2 = sess.run(weights['wd2'])
#     out_w = sess.run(weights['out'])
#     bc1 = sess.run(biases['bc1'])
#     bc2 = sess.run(biases['bc2'])
#     bc3 = sess.run(biases['bc3'])
#     bd1 = sess.run(biases['bd1'])
#     bd2 = sess.run(biases['bd2'])
#     out_b = sess.run(biases['out'])
#     FileName = 'C:/Users/alrabm/PycharmProjects/1stTFproject/Parameters.npz'
#     # f = open(FileName,'w+')
#     np.savez(FileName, wc1 = wc1,
#                 wc2 = wc2,
#                 wc3 = wc3,
#                 wd1 = wd1,
#                 wd2 = wd2,
#                 out_w = out_w,
#                 bc1 = bc1,
#                 bc2 = bc2,
#                 bc3 = bc3,
#                 bd1 = bd1,
#                 bd2 = bd2,
#                 out_b = out_b)
#
# print('Training time = %s' %( (time.time() - start_time)/60 ))


# w1 = weights['wc1']
    # print("w1 is")
    # print(sess.run(w1))

    # w1 = tf.cast(w1, tf.float32)
    # w1 = tf.multiply(10.0, w1)
    # w1 = tf.cast(w1, tf.int32)
    # w1 = tf.cast(w1, tf.float32)
    # w1 = tf.multiply(0.1, w1)
    # # w1 = tf.nn.dropout(w1, 0.1)
    # # w1 = np.array(w1)
    # print(sess.run(w1))