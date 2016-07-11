import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])
# x_data = xy[0:3]
# y_data = xy[3:]

print x_data
print y_data

# tf Graph Input
X = tf.placeholder("float", [None, 3]) # x1, x2 and 1 (for bias)
Y = tf.placeholder("float", [None, 3]) # A, B, C => 3 classes

# Set model weights
W = tf.Variable(tf.zeros([3, 3]))

# Construct Model
hypothesis = tf.nn.softmax(tf.matmul(W, X)) # Softmax

# Minimize error using cross entropy
learning_rate = 0.001

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variable
init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session() as sess:
    sess.run(init)

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)
