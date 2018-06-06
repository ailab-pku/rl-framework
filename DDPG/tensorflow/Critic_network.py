import numpy as np
import tensorflow as tf
from math import sqrt


class Critic_network():

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
        self.layer_number = 3
        self.units_number = [400, 100, 300]

        self.output, self.T_output, self.input_states, self.input_actions, self.y, self.train_op = None, None, None, None, None, None 

        with tf.name_scope('input'):
            self.input_states  = tf.placeholder('float32',[None, self.state_size],'input_state')
            self.input_actions = tf.placeholder('float32',[None, self.action_size],'input_actions')
            self.y = tf.placeholder('float32',[None, 1], name = 'y')

        with tf.variable_scope('C/source'):
            self.output = self.create_S_net()
        with tf.variable_scope('C/source_train'):
            self.create_loss_and_train_of_source()
        with tf.variable_scope('C/target'):
            self.T_output = self.create_T_net()
        with tf.variable_scope('C/target_train'):
            self.create_train_of_target()
    def create_S_net(self):

        h = [None for i in range(self.layer_number)]

        h[0] = tf.layers.dense(
            inputs = self.input_states,
            units = self.units_number[0],
            activation = tf.nn.relu,
            #kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.3, seed = 233),
            kernel_initializer=tf.random_uniform_initializer(minval=-1/sqrt(self.units_number[0]), maxval=1/sqrt(self.units_number[0])),
            #bias_initializer = tf.constant_initializer(0.1),
            #kernel_regularizer = tf.nn.l2_loss,
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg),
            name = 'h0'
            )


        h[1] = tf.layers.dense(self.input_actions,
            self.units_number[1],
            tf.nn.relu,
            kernel_initializer=tf.random_uniform_initializer(minval=-1/sqrt(self.units_number[1]), maxval=1/sqrt(self.units_number[1])),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg),                        
            name = 'h1')

        h_concat = tf.concat([h[0],h[1]],1,name = 'h_concat')

        h[2] = tf.layers.dense(h_concat,
            self.units_number[2],
            tf.nn.relu,
            kernel_initializer=tf.random_uniform_initializer(minval=-1/sqrt(self.units_number[2]), maxval=1/sqrt(self.units_number[2])),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg),                         
            name = 'h2')

        output = tf.layers.dense(h[2],
            1,
            activation = None,
            kernel_initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
            #bias_initializer = tf.constant_initializer(0.1),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg),
            name = 'output')

        return output

    def create_loss_and_train_of_source(self):

        with tf.name_scope('loss'):

            loss_function = tf.losses.mean_squared_error(self.y, self.output)#/self.batch_size
            #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #loss_function += sum(reg_losses) * 0.01   #regularization here

        with tf.name_scope('train'):
            
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_function)

        source_train_vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'C/source_train')
        source_train_vars_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope + '/C/source_train'))

        self.sess.run(tf.variables_initializer(source_train_vars_list))

    def create_train_of_target(self):

        self.target_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'C/target')
        self.source_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'C/source')

        self.op_list = [self.target_vars_list[i].assign(self.target_vars_list[i] * (1 - self.tau) + self.tau * self.source_vars_list[i]) for i in range(len(self.target_vars_list))]
        
        self.a_gradient = tf.gradients(self.output, self.input_actions)

        
    def create_T_net(self):

        source_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'C/source')
        self.sess.run(tf.variables_initializer(source_vars_list))
        output = self.create_S_net()
        target_vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'C/target')
        target_vars_op_list = [target_vars_list[i].assign(source_vars_list[i]) for i in range(len(target_vars_list))]
        self.sess.run(target_vars_op_list)

        return output

    def predict_reward_with_target_net(self, s, a):

        x = self.sess.run(self.T_output, {self.input_states: s,self.input_actions: a}) 
        return x

    def learn_with_source_net(self, s, a, y):

        self.sess.run(self.train_op, {self.input_states: s, self.input_actions: a, self.y: y})

    def learn_with_target_net(self):
        
        #print('debug',len(self.target_vars_list))
        for i in range(len(self.target_vars_list)):
            self.sess.run(self.op_list[i])

    def get_gradient_of_a_with_source_net(self, s, a):

        return np.array(self.sess.run(self.a_gradient, {self.input_states: s, self.input_actions: a}))