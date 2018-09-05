#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

x = tf.placeholder(tf.float32, shape = (None, 784))
y = tf.placeholder(tf.float32, shape = (None, 10))

# 神經網路第一層
layer_1_weight = tf.Variable(tf.random_uniform(shape = (784, 350)))
layer_1_bias = tf.Variable(tf.random_uniform(shape = (350, )))

layer_1_output = tf.nn.relu(tf.matmul(x, layer_1_weight) + layer_1_bias)

# 神經網路第二層
layer_2_weight = tf.Variable(tf.random_uniform(shape = (350, 100)))
layer_2_bias = tf.Variable(tf.random_uniform(shape = (100, )))

layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, layer_2_weight) + layer_2_bias)

# 神經網路第三層
layer_3_weight = tf.Variable(tf.random_uniform(shape = (100, 10)))
layer_3_bias = tf.Variable(tf.random_uniform(shape = (10, )))

layer_3_output = tf.matmul(layer_2_output, layer_3_weight) + layer_3_bias

# 定義 loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = layer_3_output))

# 定義 accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(layer_3_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 定義 optimizer 以及 learning rate
train_step = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 訓練 4000 次
for i in range(4000):

    # 每次訓練隨機選 200 筆資料放進神經網路
    x_batch, y_batch = mnist.train.next_batch(200)
    sess.run(train_step, feed_dict = {x: x_batch, y: y_batch})

    # 每訓練 100 次查看一次目前的 loss
    if i % 100 == 0:
        print('Step %d training loss %.6f' % (i + 1, sess.run(loss, feed_dict = {x: mnist.train.images, y: mnist.train.labels})))

# 查看神經網路訓練完後，在測試資料的準確率
print('=================')
print('Test accuracy: %.6f' % sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
