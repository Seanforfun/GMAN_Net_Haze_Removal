import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    Fetch data
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);
x = tf.placeholder(tf.float32, [None, 784]);
y_ = tf.placeholder(tf.float32, [None, 10]);
session = tf.Session();
'''

    Define our model

#x: a placeholder used to represent all graphs
#W: matrix used to save the weight of every pixel
#b: bias added to results
x = tf.placeholder(tf.float32, [None, 784]);
W = tf.Variable(tf.zeros([784, 10]));
b = tf.Variable(tf.zeros([10]));
y = tf.nn.softmax(tf.matmul(x, W) + b);

#Define our cost function by using cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10]);
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]));

#Define the method that we want our model optimize to.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy);


    Training

init = tf.global_variables_initializer();
session = tf.Session();
session.run(init);

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10);
    session.run(train_step, feed_dict={x : batch_xs, y_ : batch_ys});



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"));


    Run model to test set

print(session.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}));
'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);
    return tf.Variable(initial);


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape );
    return tf.Variable(initial);

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME');

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1 ,2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');

#Level1 layer
W_conv1 = weight_variable([5, 5, 1, 32]);
b_conv1 = bias_variable([32]);

x_image = tf.reshape(x, shape=[-1, 28, 28, 1]);
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1);
h_pool1 = max_pool_2x2(h_conv1);

#Level2 layer
W_conv2 = weight_variable([5, 5, 32, 64]);
b_conv2 = bias_variable([64]);
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2);
h_pool2 = max_pool_2x2(h_conv2);

#Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#drop out layer
keep_prob = tf.placeholder("float");
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);

#Output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2);

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))#熵的定义，与y相关
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#使用ADAM方法最小化熵
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#预测结果与真实值的一致性，这里产生的是一个bool类型的向量
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#先将bool类型转换为float类型，然后求平均值，即正确的比例
session.run(tf.initialize_all_variables())#初始化所有变量
for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
          train_accuracy = accuracy.eval(session=session, feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
          print( "step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(session = session, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print ("test accuracy %g"%accuracy.eval(session=session, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
