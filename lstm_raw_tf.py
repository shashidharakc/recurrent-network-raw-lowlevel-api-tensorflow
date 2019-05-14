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

        Ct = tf.sigmoid(tf.add(tf.multiply(Ft, previous_c_state), tf.multiply(It, CDasht)))
        current_hidden_state = tf.multiply(tf.tanh(Ct), Ot)
        current_state = tf.stack([Ct, current_hidden_state])
        
        return current_state
    
    def linear_layer(self, hidden_state):
        '''Linear layer computation'''
        return tf.add(tf.matmul(hidden_state, self.Wl), self.bl)

# Import data 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)

# Constants 
EPOCHS = 10000
INPUT_SIZE = 28
TIME_STEPS = 28
NUM_CLASSES = 10
BATCH_SIZE = 128
HIDDEN_LAYER_SIZE = 128

# Initialize RNN class 
lstm = lstm_cell(INPUT_SIZE, HIDDEN_LAYER_SIZE, NUM_CLASSES, TIME_STEPS)

# Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32,shape=[None, TIME_STEPS, INPUT_SIZE], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='labels')

# Training data set in required batch size
batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
# Reshape data to get 28 sequences of 28 pixels
batch_x = batch_x.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))

# Transpose the input data, tensorflow scan interates on the first dimension of the input data.
# Initialize hidden state for lstm, we have two inner states here 'h' and 'c' 
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
initial_hidden = tf.stack([tf.zeros([BATCH_SIZE, HIDDEN_LAYER_SIZE]), tf.zeros([BATCH_SIZE, HIDDEN_LAYER_SIZE])])

# Compute states for all lstm steps 
all_hidden_states = tf.scan(lstm.lstm, elems=processed_input, initializer=initial_hidden, name='states')
# We need only the hidden state of 'h'
all_hidden_states_transposed = tf.transpose(all_hidden_states, perm=[1,0,2,3])
all_hidden_states_c, all_hidden_states_h = tf.unstack(all_hidden_states_transposed, axis=0)

# Compute linear layer 
all_outputs = tf.map_fn(lstm.linear_layer, all_hidden_states_h)
# We need only the final layer output
output = all_outputs[-1]

# Compute loss (Here we are doing cross_entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
# Training using Adam optimizer 
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
# Compute prediction 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(output,1))
# Compute accuracy
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

# Get test dataset
test_data = mnist.test.images[:BATCH_SIZE].reshape((-1, TIME_STEPS, INPUT_SIZE))
test_label = mnist.test.labels[:BATCH_SIZE]

with tf.Session() as sess:
    # Initialize variables, train the model. 
    # Display the metrics like accuracy and loss for discreate time 
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x = batch_x.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE))
            _ = sess.run(train_step, feed_dict={_inputs:batch_x, y:batch_y})
            
            if i % 1000 == 0:
                acc,loss, = sess.run([accuracy,cross_entropy],
                                     feed_dict={_inputs: batch_x, y: batch_y})
                print ("Epoch " + str(i) + ", Minibatch Loss= " +                       "{:.3f}".format(loss) + ", Minibach Accuracy= " +                       "{:.3f}".format(acc))   
            if i % 10:
                acc = sess.run( accuracy, feed_dict={_inputs: test_data, y: test_label})

    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})
    print ("Test Accuracy:", test_acc)


