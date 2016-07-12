import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

training_epochs = 25
batch_size = 100
display_step = 1

# xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
# x_data = np.transpose(xy[0:3])
# y_data = np.transpose(xy[3:])

# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes


# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct Model
# https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginnners/index.html
# First, we multiply x by W with the expression tf.matmul(x, W).
# This is flipped from when we multiplied them in our equation,
# where we had Wx, as a small trick to deal with x being
# a 2D tensor with multiple inputs.
# We then add b, and finally apply tf.nn.softmax
#hypothesis = tf.nn.softmax(tf.matmul(X, W)) # Softmax
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
learning_rate = 0.001
# learning_rate = 10

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variable
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Get one and predict
    r = randint(0, mnist.test.num_examples - 1)
    print "Label: ", sess.run(tf.arg_max(mnist.test.labels[r:r+1], 1))
    print "Prediction: ", sess.run(tf.arg_max(activation, 1), {x: mnist.test.images[r:r+1]})

    # Show the Image
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()