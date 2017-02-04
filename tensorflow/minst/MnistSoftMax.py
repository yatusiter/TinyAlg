# coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

sess    = tf.InteractiveSession()

x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W  = tf.Variable(tf.zeros([784,10]))
b  = tf.Variable(tf.zeros([10]))

y  = tf.matmul(x, W) + b

crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
trainStep    = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    trainStep.run(feed_dict={x: batch[0], y_:batch[1]})

correctPrediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

print accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
