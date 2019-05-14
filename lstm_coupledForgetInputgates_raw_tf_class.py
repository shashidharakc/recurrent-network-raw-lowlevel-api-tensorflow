#!/usr/bin/env python
# coding: utf-8

#Import dependencies 
import numpy as np
import tensorflow as tf

class lstm_cell(object):

    def __init__(self, input_size, hidden_layer_size, target_size, time_steps):
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
        self.time_steps = time_steps
        
        # Weights initialization Zt Weights 
        with tf.name_scope('rnn_weights_It'):
            self.Wx_i = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh_i = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))

        with tf.name_scope('rnn_weights_Ft'):
            self.Wx_f = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh_f = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))

        with tf.name_scope('rnn_weights_Ot'):
            self.Wx_o = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh_o = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))

        with tf.name_scope('rnn_weights_Gt'):
            self.Wx_g = tf.Variable(tf.zeros([self.input_size, self.hidden_layer_size]))
            self.Wh_g = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))
        # Initialize dense layer(output) weights
        with tf.name_scope('linear_layer_weights'):
            self.Wl = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size ], mean=0,stddev=.01))
            self.bl = tf.Variable(tf.truncated_normal([self.target_size], mean=0,stddev=.01))

    def lstm(self, previous_state, x):
        '''lstm computation for single step'''
        previous_c_state, previous_hidden_state = tf.unstack(previous_state, axis=0)

        It = tf.sigmoid(tf.add(tf.matmul(x, self.Wx_i),tf.matmul(self.Wh_i, previous_hidden_state)))
        Ft = tf.sigmoid(tf.add(tf.matmul(x, self.Wx_f),tf.matmul(self.Wh_f, previous_hidden_state)))
        Ot = tf.sigmoid(tf.add(tf.matmul(x, self.Wx_o),tf.matmul(self.Wh_o, previous_hidden_state)))
        CDasht = tf.tanh(tf.add(tf.matmul(x, self.Wx_g),tf.matmul(self.Wh_g, previous_hidden_state)))

        Ct = tf.sigmoid(tf.add(tf.multiply(Ft, previous_c_state), tf.multiply(1-Ft, CDasht)))
        current_hidden_state = tf.multiply(tf.tanh(Ct), Ot)
        current_state = tf.stack([Ct, current_hidden_state])
        
        return current_state
    
    def linear_layer(self, hidden_state):
        '''Linear layer computation'''
        return tf.add(tf.matmul(hidden_state, self.Wl), self.bl)
