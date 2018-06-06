import numpy as np
import tensorflow as tf
from math import sqrt
class Actor_network():

    def __init__(self, sess, scope,
                 state_size,
                 action_size,
                 batch_size,
                 learning_rate,
                 tau):
        
        self.batch_size = batch_size
        self.sess = sess
        self.scope = scope
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.l2_reg = 0.01
        self.layer_number = 2
        self.units_number = [400, 300]

        self.output, self.T_output, self.input_states, self.input_gradient, self.train_op = None, None, None, None, None

       
        with tf.name_scope('input'):
            self.input_states  = tf.placeholder('float32',[None, self.state_size],'input_state2')
            self.input_gradient = tf.placeholder('float32',[None, self.action_size],'input_gradient')
        
        with tf.variable_scope('A/source'):
            self.output = self.create_S_net()
        with tf.variable_scope('A/source_train'):
            self.create_loss_and_train_of_source()
        with tf.variable_scope('A/target'):
            self.T_output = self.create_T_net()
        with tf.variable_scope('A/target_train'):
            self.create_train_of_target()

        
    def create_S_net(self):

        h = [None for i in range(self.layer_number)]

        h[0] = tf.layers.dense(self.input_states,
            self.units_number[0],
            tf.nn.relu,
            kernel_initializer = tf.random_uniform_initializer(minval=-1/sqrt(self.units_number[0]), maxval=1/sqrt(self.units_number[0])),
            #bias_initializer = tf.constant_initializer(0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), 
            name = 'h0')

        h[1] = tf.layers.dense(h[0],
            self.units_number[1],
            tf.nn.relu,
            kernel_initializer = tf.random_uniform_initializer(minval=-1/sqrt(self.units_number[1]), maxval=1/sqrt(self.units_number[1])),
            #bias_initializer = tf.constant_initializer(0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),               
            name = 'h1')

                
        output = tf.layers.dense(h[1],
            self.action_size,
            tf.nn.tanh,
            kernel_initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
            #bias_initializer = tf.constant_initializer(0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),                                    
            name = 'output')

        return output

    def create_T_net(self):

        source_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'A/source')
        self.sess.run(tf.variables_initializer(source_vars_list))

        output = self.create_S_net()
        target_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'A/target')
        target_vars_op_list = [target_vars_list[i].assign(source_vars_list[i]) for i in range(len(target_vars_list))]
        self.sess.run(target_vars_op_list)

        return output

    def choose_action_with_target_net(self, s):
        x = self.sess.run(self.T_output, feed_dict = {self.input_states: s}) 
        return x

    def choose_action_with_source_net(self, s):

        x = self.sess.run(self.output, feed_dict = {self.input_states: s})
        return x

    def create_loss_and_train_of_source(self):

        with tf.name_scope('loss'):
                    
            source_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'A/source')
            gradient = tf.gradients(self.output, source_vars_list, - self.input_gradient)

        with tf.name_scope('train'):
            
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(gradient, source_vars_list),global_step=tf.contrib.framework.get_global_step())

        source_train_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'A/source_train')
        source_train_vars_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope + '/A/source_train'))
        
        self.sess.run(tf.variables_initializer(source_train_vars_list))

    def create_train_of_target(self):

        self.target_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'A/target')
        self.source_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'A/source')

        self.op_list = [self.target_vars_list[i].assign(self.target_vars_list[i] * (1 - self.tau) + self.tau * self.source_vars_list[i]) for i in range(len(self.target_vars_list))]


    def learn_with_source_net(self, s, a_source_gradient):

        batch_size = len(a_source_gradient)
        self.sess.run(self.train_op, {self.input_states: s, self.input_gradient: a_source_gradient / batch_size})

    def learn_with_target_net(self):
        for i in range(len(self.target_vars_list)):
            self.sess.run(self.op_list[i])

            




