"""
Convolutional layers
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_func=None):
    """
        Creates hidden layers
    """
    weights = tf.Variable(tf.random_normal([in_size, out_size])) # initialize all variables as 0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases

    if activation_func:
        return activation_func(wx_plus_b)
    return wx_plus_b

# Create some data
X = np.linspace(-1, 1, 300)[:, np.newaxis] # add 1D to sample
noise = np.random.normal(0, 0.05, X.shape).astype(np.float32) # add some noise so it's looks real
# y = x^2
Y = np.square(X) - 0.5 + noise

# plot the data
#plt.scatter(X, Y)
#plt.show()

# Define inputs (placeholders)
xs = tf.placeholder(tf.float32, [None, 1]) # n samples, n futures
ys = tf.placeholder(tf.float32, [None, 1])

# Add hidden layer
l1 = add_layer(xs, 1, 10, activation_func=tf.nn.relu) # data, future size, hidden neuron, actv_fnc

# Add output layer
prediction = add_layer(l1, 10, 1, activation_func=None) # data, future size, expectation size, actv_fnc

# Calculate errors
# calculate the square errors (difference between real and predicted values) and sum all of them
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# Train
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X, Y)
plt.ion()
plt.show()
for i in range(1000):
    # egitim sart
    sess.run(train_step, feed_dict = {xs:X, ys:Y})
    
    if i % 50 == 0:
        # loss is based on ys and prediction and prediction is based on xs 
        #print(sess.run(loss, feed_dict = {xs:X, ys:Y}))
        
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction, feed_dict={xs:X})    
        # plot
        lines = ax.plot(X, prediction_value, 'r', lw=5)
        plt.pause(1)
