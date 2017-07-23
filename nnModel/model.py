import tensorflow as tf
import numpy as np
def build_model(width):
    with tf.name_scope('model'):
        grasp_image_ph = tf.placeholder('float', [None, width, width, 3])
        keep_prob_ph = tf.placeholder('float', name='dropout')

        # rgb conv
        a1 = tf.contrib.layers.convolution2d(tf.nn.dropout(grasp_image_ph, keep_prob_ph), 32, (5, 5), activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope='conv1')

        a2 = tf.contrib.layers.convolution2d(tf.nn.dropout(a1, keep_prob_ph), 64, (3, 3), activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope='conv2')
        a2_max = tf.nn.max_pool(a2, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME')
        a3 = tf.contrib.layers.convolution2d(tf.nn.dropout(a2_max, keep_prob_ph), 64, (3, 3), activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope='conv3')
        a3_max = tf.nn.max_pool(a3, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME')
        a4 = tf.contrib.layers.convolution2d(tf.nn.dropout(a3_max, keep_prob_ph), 32, (3, 3), activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope='conv4')
        a4_max = tf.nn.max_pool(a4, (1, 3, 3, 1), (1, 2, 2, 1), 'SAME')

        conv = a4_max

        # flatten
        conv_shape = conv.get_shape().as_list()
        flat_dim = np.product(conv_shape[1:])
        print ('Final shape', conv_shape, 'flat_dim', flat_dim)
        conv_flat = tf.reshape(conv, [-1, flat_dim])

        # fc
        fc1 = tf.contrib.layers.fully_connected(conv_flat, 2, weights_initializer=tf.contrib.layers.xavier_initializer(), scope='fc1')

        # prediction
        logit = fc1
        grasp_class_prediction = tf.nn.softmax(fc1)
        # depth_prediction = d2

        # return grasp_class_prediction, logit, grasp_image_ph, keep_prob_ph
        return grasp_class_prediction, logit, grasp_image_ph, keep_prob_ph