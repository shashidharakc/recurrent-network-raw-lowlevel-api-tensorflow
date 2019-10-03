#!/usr/bin/env python
# coding: utf-8

#Import dependencies 
import numpy as np
import tensorflow as tf

class LSTM_Peephole_build(tf.keras.layers.Layer):
    """
    This is a class for computing Peephole LSTM, as described in colah's blog.
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    Embedding size of the input data found out after running the first input. 
    
    Forward Computation : 
    Ft = sigmoid(Wcf*Ct^-1 + Whf*Ht^-1 + Wxf*Xt)
    It = sigmoid(Wci*Ct^-1 + Whi*Ht^-1 + Wxi*Xt)
    Ct_Dash = tanh(Whc*Ht^-1 + Wxc*Xt)
    Ct = Ft*Ct^-1 + It*Ct_Dash
    Ot = sigmoid(Wco*Ct +Who*Ht^-1 + Wxo*Xt)
    Ht = Ot*tanh(Ct)    
    
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
        super(LSTM_Peephole_build, self).__init__()
        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_UNITS = HIDDEN_UNITS
        self.VOCAB_SIZE = VOCAB_SIZE
    
        self.w_forget = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_input = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_output = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_c_dash = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        
        self.w_forget_c = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_input_c = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_output_c = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        
        self.w_op = tf.Variable(initial_value=w_init(shape=(self.HIDDEN_UNITS, self.VOCAB_SIZE), dtype='float32'), trainable=True)
        
    def build(self, input_shape):
        """Initiaize the weights of params, after the first pass"""
        w_init = tf.random_normal_initializer()
        self.EMBEDDING_SIZE = input_shape[-1]
        self.w_ip_f = tf.Variable(initial_value=w_init(shape=(self.EMBEDDING_SIZE, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_ip_i = tf.Variable(initial_value=w_init(shape=(self.EMBEDDING_SIZE, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_ip_o = tf.Variable(initial_value=w_init(shape=(self.EMBEDDING_SIZE, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        self.w_ip_c_dash = tf.Variable(initial_value=w_init(shape=(self.EMBEDDING_SIZE, self.HIDDEN_UNITS), dtype='float32'), trainable=True)
        
    def call(self, input_data, previous_hidden_state,previous_c_state):
        """Define forward pass"""
        Ft = tf.sigmoid(tf.matmul(input_data, self.w_ip_f) + tf.matmul(previous_hidden_state, self.w_forget) + tf.matmul(previous_c_state, self.w_forget_c))
        It = tf.sigmoid(tf.matmul(input_data, self.w_ip_i) + tf.matmul(previous_hidden_state, self.w_input) + tf.matmul(previous_c_state, self.w_input_c))
        Ct_dash = tf.tanh(tf.matmul(input_data, self.w_ip_c_dash) + tf.matmul(previous_hidden_state, self.w_c_dash))
        Ct = Ft*previous_c_state + It*Ct_dash
        Ot = tf.sigmoid(tf.matmul(input_data, self.w_ip_o) + tf.matmul(previous_hidden_state, self.w_output) + tf.matmul(Ct, self.w_output_c))

        Ht = Ot*tf.tanh(Ct)
        Yt = tf.sigmoid(tf.matmul(Ht, self.w_op))
        return Yt, Ht, Ct
        
