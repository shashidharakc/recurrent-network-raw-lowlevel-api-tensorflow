#!/usr/bin/env python
# coding: utf-8

#Import dependencies 
import numpy as np
import tensorflow as tf

class vanilla_rnn_cell(object):

    def __init__(self, input_size, hidden_layer_size, target_size, time_steps):
    
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
        # Initialize rnn weights
        with tf.name_scope('rnn_weights_Zt'):
            self.Wx = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
            self.b = tf.Variable(tf.zeros([self.hidden_layer_size])) 
        # Initialize dense layer(output) weights
        with tf.name_scope('linear_layer_weights'):
            self.Wl = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size ], mean=0,stddev=.01))
            self.bl = tf.Variable(tf.truncated_normal([self.target_size], mean=0,stddev=.01))

    def rnn(self, previous_hidden_state, x):
        '''RNN computation for single step'''
        current_hidden_state = tf.tanh(tf.add(tf.add(
            tf.matmul(previous_hidden_state, self.Wh), tf.matmul(x, self.Wx)), self.b))

        return current_hidden_state

    def linear_layer(self, hidden_state):
        '''Linear layer computation'''
        return tf.add(tf.matmul(hidden_state, self.Wl), self.bl)
