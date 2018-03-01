#=====================================================================#
# Example 1
#=====================================================================#

# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow as tf
#
#
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# sess = tf.Session()
#
# x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
#
# W = tf.Variable(tf.ones([784,10]))
# b = tf.Variable(tf.ones([10]))
#
# sess.run(tf.global_variables_initializer())
#
# print(sess.run(W))
#
# y = tf.matmul(x,W) + b
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# for ii in range(1000):
#   batch = mnist.train.next_batch(100)
#   #train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#   sess.run( train_step, feed_dict={x: batch[0], y_: batch[1]} )
#   print('After training W =', sess.run(W) )
#
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Out = sess.run( accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels} )
# print(Out)

#=================================================================================#
# EXAMPLE 2
#=================================================================================#
# import tensorflow as tf
#
# # Model parameters
# W = tf.Variable([.3], dtype=tf.float32)
# b = tf.Variable([-.3], dtype=tf.float32)
# # Model input and output
# x = tf.placeholder(tf.float32)
# linear_model = W*x + b
# y = tf.placeholder(tf.float32)
#
# # loss
# loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# # optimizer
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# # training data
# x_train = [1, 2, 3, 4]
# y_train = [0, -1, -2, -3]
# # training loop
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init) # reset values to wrong
# for i in range(1000):
#   sess.run(train, {x: x_train, y: y_train})
#
# # evaluate training accuracy
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#==================================================================#
# Example 3
#==================================================================#
# import tensorflow as tf
#
# lr = 0.01
# X = tf.placeholder(tf.float32,shape=[],name='Input')
# Y = tf.placeholder(tf.float32,shape=[],name='Label')
# w1 = tf.Variable(tf.random_normal([],mean=1,stddev=2),dtype=tf.float32,name='w1')
# w2 = tf.Variable(tf.random_normal([],mean=1,stddev=2),dtype=tf.float32,name='w2')
# w3 = tf.Variable(tf.random_normal([],mean=1,stddev=2),dtype=tf.float32,name='w3')
# y1 = tf.multiply(X,w1,name='y1')
# y1 = tf.stop_gradient(y1)
# y2 = tf.multiply(y1,w2,name='y2')
# y3 = tf.multiply(y2,w3,name='Output')
#
# init = tf.global_variables_initializer()
# loss_func = tf.square( tf.subtract(y3,Y) )
# optimizer = tf.train.GradientDescentOptimizer(lr)
# training = optimizer.minimize(loss_func)
#
# x = [1, 3, 2.4, 4, 8]
# yLabel = [0.3, 0.4, 7, 10, 5]
#
# with tf.Session() as sess:
#     sess.run(init)
#     print('Before training')
#     print('w1 = %s' %sess.run(w1))
#     print('w2 = %s' %sess.run(w2))
#     print('w3 = %s' %sess.run(w3))
#     for ii in range(4):
#         feed_dict = {X:x[ii],Y:yLabel[ii]}
#         sess.run(training,feed_dict)
#     print('After training')
#     print('w1 = %s' %sess.run(w1))
#     print('w2 = %s' %sess.run(w2))
#     print('w3 = %s' %sess.run(w3))
#====================================================================#
# Example 4
#====================================================================#
# import numpy as np
#
# a = np.linspace(0.001, 1.1, num=100)
# for ii in np.nditer(a):
#     print(ii)
#    print(ii**2)
#====================================================================#
# Example 5
#====================================================================#
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data # Imports the MINST dataset

# Data Set:
# ---------
mnist = input_data.read_data_sets("/home/frr/MNIST_data", one_hot=True)# An object where data is stored

# Learning Parameters:
# --------------------
learning_rate = 0.0001
training_iters = 10000
batch_size = 64
display_step = 20
ImVecDim = 784# The number of elements in a an image vector (flattening a 28x28 2D image)
NumOfClasses = 10
dropout = 0.8

# w_mod = tf.Variable( tf.random_normal([3, 3, 64, 128]), name='mod_weights2', trainable=False )
g = tf.get_default_graph()

saved = np.load('Parameters.npz')
print(saved['wc1'])
with tf.Session() as sess:
    LoadMod = tf.train.import_meta_graph('simple_mnist.ckpt.meta')  # This object loads the model
    LoadMod.restore(sess, tf.train.latest_checkpoint('./'))  # Loading weights and biases and other stuff to the model

    # Printing parameters to a text file:
    wc1 = g.get_tensor_by_name('wc1:0')
    bc1 = g.get_tensor_by_name('bc1:0')
    wc2 = g.get_tensor_by_name('wc2:0')
    bc2 = g.get_tensor_by_name('bc2:0')
    wc3 = g.get_tensor_by_name('wc3:0')
    bc3 = g.get_tensor_by_name('bc3:0')
    wd1 = g.get_tensor_by_name('wd1:0')
    bd1 = g.get_tensor_by_name('bd1:0')
    wd2 = g.get_tensor_by_name('wd2:0')
    bd2 = g.get_tensor_by_name('bd2:0')
    out_w = g.get_tensor_by_name('out_w:0')
    out_b = g.get_tensor_by_name('out_b:0')
    WeightList = [wc1,wc2,wc3,wd1,wd2,out_w]
    BiasList = [bc1,bc2,bc3,bd1,bd2,out_b]

    # for ii in range(6):
    #     filePath = './Weights' + str(ii+1) + '.txt'
    #     print(filePath)
    #     f1 = open(filePath,'w+')
    #     X = sess.run(WeightList[ii])
    #     Dim = X.shape
    #     print(Dim)
    #     if len(Dim) == 4:
    #         for Filt in range(Dim[3]):
    #             for ch in range(Dim[2]):
    #                 for row in range(Dim[0]):
    #                     for col in range(Dim[1]):
    #                         f1.write('%s ' %X[row,col,ch,Filt])
    #                     f1.write('\n')
    #     if len(Dim) == 2:
    #         for row in range(Dim[0]):
    #             for col in range(Dim[1]):
    #                 f1.write('%s ' % X[row, col])
    #             f1.write('\n')
    #     else:
    #         print('Dimensions problem')

        # filePath = './Biases' + str(ii + 1) + '.txt'
        # print(filePath)
        # f2 = open(filePath, 'w+')
        # X = sess.run(BiasList[ii])
        # Dim = X.shape
        # print(Dim)
        # for col in range(Dim[0]):
        #     f2.write('%s ' %X[col])



    x = g.get_tensor_by_name('ImageIn:0')
    y = g.get_tensor_by_name('LabelIn:0')
    keep_prob = g.get_tensor_by_name('KeepProb:0')
    cost = g.get_tensor_by_name('loss:0')
    accuracy = g.get_tensor_by_name('NetAccuracy:0')
    VarToTrain = g.get_collection_ref('trainable_variables')
    print('--------------------------------------------------------')
    print(VarToTrain)
    del VarToTrain[0]
    print(VarToTrain)
    del VarToTrain[5]
    print(VarToTrain)
    # print(VarToTrain[1])
    # print(VarToTrain[7])
    # del VarToTrain[0]
    # del VarToTrain[5]
    # print( VarToTrain )
    # print(g.get_all_collection_keys())
    # print(g.get_collection('train_op'))
    # print(g.get_collection('trainable_variables'))
    # print(g.get_collection('variables'))


    # optimizer = g.get_operation_by_name('AdamOpt')
    # print(optimizer)
    # optimizer().minimize(cost,var_list=VarToTrain)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='NewAdam').\
    #     minimize(cost,var_list = [VarToTrain[1], VarToTrain[7]])
    # print(g.get_collection('train_op'))
    # step = 1
    # while step * batch_size < training_iters:
    #     print('commence training')
    #     batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #     sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
    #     if step % display_step == 0:
    #         acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
    #         #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
    #         print("Iter " + str(step*batch_size) + ", Training Accuracy= " + "{:.5f}".format(acc))
    #     step += 1
    # print("Optimization Finished!")
    # print("Testing Accuracy:",
    # sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
    #
    # # Checkpoint
    # print('After retraining')
    # wc1 = sess.run(g.get_tensor_by_name('wc1:0'))
    # wc2 = sess.run(g.get_tensor_by_name('wc2:0'))
    # print('Sample of wc1 = %f' % wc1[0, 0, 0, 0])
    # print('Sample of wc2 = %f' % wc2[0, 0, 0, 0])


    # op = sess.graph.get_operations()
    # print( [m.values() for m in op] )