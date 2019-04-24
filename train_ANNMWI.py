"""
Implementation of training ANN I or ANN II on generating MWI.

Created with Tensorflow 1.7.0 and Python 2.7 using CUDA 8.0.

Copyright @ Jieun Lee
Laboratory for Imaging Science and Technology
Seoul National University
jjje0924@gmail.com
"""

import tensorflow as tf
import numpy as np
import os
import h5py
import time
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load data
load = h5py.File('../(change_folder_name)/processed_train.mat')
x_train = load['train']
label_train = load['target']
x_val = load['valid']
label_val = load['valid_target']

# Feed in array
x_train = x_train[:,:]
label_train = label_train[:,:]
x_val = x_val[:,:]
label_val = label_val[:,:]

# Check the shape
print(x_train.shape)
print(label_train.shape)
print(x_val.shape)
print(label_val.shape)
print('Data Loaded')

# Hyper-parameters
init_learning_rate = 0.001
training_epochs = 2000
batch_size = 2
display_step = 1
save_step = 1000
valid_step = 100

tf.reset_default_graph()

# Hidden layers & neurons
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
    return (out_layer)

# tf Graph input
X = tf.placeholder("float", [None, n_input], name="inputX")
Y = tf.placeholder("float", [None, n_classes], name="labelY")
Z = tf.placeholder("float", [None, n_input], name="valX")
U = tf.placeholder("float", [None, n_classes], name="valY")
learning_rate = tf.placeholder(tf.float32, name="learning_rate")

# L2 loss function
def l2(f, d):
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(f - d), 1) + 1e-8))
    return loss

# Construct tf model for training
logits = multilayer_perceptron(X, False)
loss_op = l2((logits), Y)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Construct tf model for validation
logits_val = multilayer_perceptron(Z, True)
loss_val = l2((logits_val), U)

# Accuracy
correct_prediction = tf.equal(tf.argmax((logits_val), 1), tf.argmax(U, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
ind = list(range(len(x_train)))

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

epoch_restore = 0

with tf.Session() as sess:

    sess.run(init)
    t = time.time()
    loss_vec = []
    loss_valid = []
    epoch_learning_rate = init_learning_rate

    # Training cycle
    print('Start training')
    for epoch in tqdm(range(training_epochs - epoch_restore)):
        random.shuffle(ind)
        avg_cost = 0.

        BS = (batch_size + epoch)
        total_batch = int(x_train.shape[0] / BS)

        if (epoch + epoch_restore) == (900) or epoch == (1200) or epoch == (1500) or epoch == (1800):
            epoch_learning_rate = epoch_learning_rate / 10

        # Loop over all batches
        for i in range(0, len(x_train) - BS, BS):
            ind2 = ind[i:i + BS]
            ind2 = np.sort(ind2)
            batch_x = x_train[ind2, :]
            batch_y = label_train[ind2, :]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y, learning_rate: epoch_learning_rate})
            # Compute average loss
            avg_cost += c / total_batch

        # Display cost per epoch step
        print("cost={:.6f}".format(avg_cost))

        # Check the validation set
        if (epoch+epoch_restore) % valid_step == valid_step - 1:
            print('Check validation...')
            acc, cost_val = sess.run([accuracy, loss_val], feed_dict={Z: x_val, U: label_val, learning_rate: epoch_learning_rate})
            print("cost_val={:.6f}".format(cost_val), "acc={:.6f}".format(acc))
            loss_vec.append(avg_cost)
            loss_valid.append(cost_val)
            # Uncomment the annotations under three lines if you want to save the output from the validation set
            #pred_val = sess.run([logits_val], feed_dict={Z: x_val, U: label_val, learning_rate: epoch_learning_rate})
            #savepath = '../(change_folder_name)/valid_out' + str((epoch+epoch_restore))
            #scipy.io.savemat(savepath + '.mat', mdict={'valid_out': pred_val})
            print('Prediction Saved!')

        # Save sess
        if (epoch+epoch_restore) % save_step == save_step - 1:
            saver_path = saver.save(sess, "../(change_folder_name)/train_result/ANN2/epoch" + str((epoch+epoch_restore)) + ".ckpt")

    print('Total training time:{:.4f}'.format(time.time() - t))

    print("Optimization Finished!")

    # Plot loss over time
    plt.plot(loss_vec, label='training loss')
    plt.plot(loss_valid, label='validation loss')
    plt.legend(loc='upper right')
    plt.show()
