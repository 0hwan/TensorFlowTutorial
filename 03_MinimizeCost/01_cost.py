import tensorflow as tf
import matplotlib.pyplot as plt

# tf Graph Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)

# Set model weights
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.mul(X, W)

# Cost Function
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)

# Before starting, initialize the variable. We will 'run' this first.
init = tf.initialize_all_variables()

# For graphs
W_val = []
cost_val = []

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for i in xrange(-30, 50):
    print i*0.1, sess.run(cost, feed_dict={W: i*0.1})
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

# Graphic display
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()

