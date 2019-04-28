"""
Implementation of the network test.

Created with Tensorflow 1.7.0 and Python 2.7 using CUDA 8.0.

Copyright @ Jieun Lee
Laboratory for Imaging Science and Technology
Seoul National University
jjje0924@gmail.com
"""

import tensorflow as tf
import os
import h5py
import time
import scipy.io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

n_input = 32
n_hidden_1 = 160
n_hidden_2 = 240
n_hidden_3 = 320
n_hidden_4 = 360
n_hidden_5 = 480
n_hidden_6 = 520
n_hidden_7 = 600
n_classes = 120   # Set 1 for ANN I / Set 120 for ANN II

# Store weight & bias
def params(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", shape=bias_shape, initializer=tf.random_normal_initializer())
    output = tf.nn.leaky_relu(tf.add(tf.matmul(input, weights), biases))
    return output

# Create neural network
def multilayer_perceptron(x, reuse):
    with tf.variable_scope("hidden1", reuse=reuse):
        layer_1 = params(x, [n_input, n_hidden_1], [n_hidden_1])
    with tf.variable_scope("hidden2", reuse=reuse):
        layer_2 = params(layer_1, [n_hidden_1, n_hidden_2], [n_hidden_2])
    with tf.variable_scope("hidden3", reuse=reuse):
        layer_3 = params(layer_2, [n_hidden_2, n_hidden_3], [n_hidden_3])
    with tf.variable_scope("hidden4", reuse=reuse):
        layer_4 = params(layer_3, [n_hidden_3, n_hidden_4], [n_hidden_4])
    with tf.variable_scope("hidden5", reuse=reuse):
        layer_5 = params(layer_4, [n_hidden_4, n_hidden_5], [n_hidden_5])
    with tf.variable_scope("hidden6", reuse=reuse):
        layer_6 = params(layer_5, [n_hidden_5, n_hidden_6], [n_hidden_6])
    with tf.variable_scope("hidden7", reuse=reuse):
        layer_7 = params(layer_6, [n_hidden_6, n_hidden_7], [n_hidden_7])
    with tf.variable_scope("outputs", reuse=reuse):
        out_layer = params(layer_7, [n_hidden_7, n_classes], [n_classes])
    return tf.maximum(out_layer,0)


f = h5py.File('/SNU/list/Jieun/processed_test.mat')
test = f["testset"]
test = test[:,:]

Z = tf.placeholder("float", [None, n_input])

result = multilayer_perceptron(Z, False)
init = tf.global_variables_initializer()
saver=tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(init)
saver.restore(sess, "../(change_folder_name)/train_result/ANN2/epoch1999.ckpt")
print("restore finished")

st = time.time()
testout = sess.run(result, feed_dict={Z: test})
print("[processing time : ] {} sec".format(time.time() - st))

scipy.io.savemat('../(change_folder_name)/train_result/ANN2/test_out.mat',
                 mdict = {'test_out': testout})

print("save finished")

# Use the code below to save the network parameters

# for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#     print i
#
# w1 = sess.run("hidden1/weights:0")
# w2 = sess.run("hidden2/weights:0")
# w3 = sess.run("hidden3/weights:0")
# w4 = sess.run("hidden4/weights:0")
# w5 = sess.run("hidden5/weights:0")
# w6 = sess.run("hidden6/weights:0")
# w7 = sess.run("hidden7/weights:0")
# w8 = sess.run("outputs/weights:0")
#
# B1 = sess.run("hidden1/biases:0")
# B2 = sess.run("hidden2/biases:0")
# B3 = sess.run("hidden3/biases:0")
# B4 = sess.run("hidden4/biases:0")
# B5 = sess.run("hidden5/biases:0")
# B6 = sess.run("hidden6/biases:0")
# B7 = sess.run("hidden7/biases:0")
# B8 = sess.run("outputs/biases:0")
#
# scipy.io.savemat('../(change_folder_name)/network_parameter',
#                  mdict={'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7':w7, 'w8':w8,
#                         'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5, 'B6': B6, 'B7':B7, 'B8':B8})
