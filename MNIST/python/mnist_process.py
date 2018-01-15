import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
    Fetch data
'''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);

'''
    Define our model
'''
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

'''
    Training
'''
init = tf.global_variables_initializer();
session = tf.Session();
session.run(init);

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100);
    session.run(train_step, feed_dict={x : batch_xs, y_ : batch_ys});

'''
    Evaluation
'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"));

'''
    Run model to test set
'''
print(session.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}));