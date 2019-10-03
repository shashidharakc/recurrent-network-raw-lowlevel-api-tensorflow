#!/usr/bin/env python
# coding: utf-8

#Import dependencies 
import numpy as np
import tensorflow as tf

class RNN_Vanila_build(tf.keras.layers.Layer):
    """
    This is a class for computing Vanilla RNN(elman rnn network), Embedding size of the input data found out after
    running the first input. 
    
    Attributes:
        BATCH_SIZE(int)     : Batch size of the data 
        HIDDEN_UNITS(int)   : # hidden units
        VOCAB_SIZE(int)     : Vocab size of the output data
    
    Note: 
        Assumed embedded data is pushed when calling the forward computation
        ouput of the each hidden layer is projected to another non linear function 
    
    """
    def __init__(self, BATCH_SIZE, HIDDEN_UNITS, VOCAB_SIZE):
        """Initialize weights for the params, which dosen't need to know input shape"""
        super(RNN_Vanila_build, self).__init__()
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_UNITS = HIDDEN_UNITS
        self.VOCAB_SIZE = VOCAB_SIZE
    
        self.w_hidden = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_op = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.VOCAB_SIZE), dtype='float32'), trainable=True)
        
    def build(self, input_shape):
        """Initiaize the weights of params, after the first pass"""
        w_init = tf.random_normal_initializer()
        self.EMBEDDING_SIZE = input_shape[-1]
        self.w_ip = tf.Variable(initial_value=w_init(shape=(self.EMBEDDING_SIZE, self.HIDDEN_UNITS), dtype='float32'), trainable=True)

        
    def call(self, input_data, previous_hidden_state):
        """Define forward pass"""
        Ht = tf.tanh(tf.matmul(input_data, self.w_ip) + tf.matmul(previous_hidden_state, self.w_hidden))
        Yt = tf.sigmoid(tf.matmul(Ht, self.w_op))
        return Yt, Ht
        
