#!/usr/bin/env python
# coding: utf-8

#Import dependencies 
import numpy as np
import tensorflow as tf

class gru_cell(object):
    
    def __init__(self, input_size, hidden_layer_size, target_size, time_steps):
    
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
        self.time_steps = time_steps
        # Initialize rnn weights
        with tf.name_scope('rnn_weights_Zt'):
            self.Wx_z = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh_z = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))

        with tf.name_scope('rnn_weights_Rt'):
            self.Wx_r = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh_r = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))

        with tf.name_scope('rnn_weights_HdashT'):
            self.Wx_hdash = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh_hdash = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
        # Initialize dense layer(output) weights
        with tf.name_scope('linear_layer_weights'):
            self.Wl = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size ], mean=0,stddev=.01))
            self.bl = tf.Variable(tf.truncated_normal([self.target_size], mean=0,stddev=.01))

    def gru(self, previous_hidden_state, x):
        '''GRU computation for single step'''
        Zt = tf.sigmoid(tf.add(tf.matmul(x, self.Wx_z),tf.matmul(self.Wh_z, previous_hidden_state)))
        Rt = tf.sigmoid(tf.add(tf.matmul(x, self.Wx_r),tf.matmul(self.Wh_r, previous_hidden_state)))
        HDash = tf.tanh(tf.add(tf.matmul(x, self.Wx_hdash), tf.multiply(Rt, tf.matmul(previous_hidden_state, self.Wh_hdash))))
        # HDash = tf.tanh(tf.add(tf.matmul(x, self.Wx_hdash), tf.matmul(tf.multiply(Rt, previous_hidden_state), self.Wh_hdash)))
        current_hidden_state = tf.add(tf.multiply((1-Zt), previous_hidden_state), tf.multiply(Zt,HDash))

        return current_hidden_state

    def linear_layer(self, hidden_state):
        '''Linear layer computation'''
        return tf.add(tf.matmul(hidden_state, self.Wl), self.bl)
