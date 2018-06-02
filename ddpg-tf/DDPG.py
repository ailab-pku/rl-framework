import numpy as np
import tensorflow as tf
from Critic_network import Critic_network
from Actor_network import Actor_network


class DDPG_model:

    def __init__(self,
                state_size,                     #state的维数
                action_size,                    #action的维数
                batch_size,                     #一组batch的数量
                gamma = 0.99,                   #RL衰减系数γ
                actor_learning_rate = 1e-4,     #actor_source_network 学习率
                critic_learning_rate = 1e-3,    #critic_source_network 学习率
                tau = 1e-3):                    #actor & critic target_network 逼近source_network的参数τ
        
        self.state_size = state_size
        self.action_size = action_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        self.C_net_scope = 'C_net'
        self.A_net_scope = 'A_net'
        self.sess = tf.Session()
        
        with tf.name_scope(self.A_net_scope):
            self.A_network = Actor_network(self.sess,
                    self.A_net_scope,
                    self.state_size,
                    self.action_size,
                    self.batch_size,
                    self.actor_learning_rate,
                    self.tau
                    )

        with tf.name_scope(self.C_net_scope):
            self.C_network = Critic_network(self.sess,
                    self.C_net_scope,
                    self.state_size,
                    self.action_size,
                    self.batch_size,
                    self.critic_learning_rate,
                    self.tau
                    )

    def choose_action(self, present_state):

        
        with tf.name_scope(self.A_net_scope):
            return self.A_network.choose_action_with_source_net([present_state])

    def learn(self, memory_buffer):

        h = memory_buffer.choose_n_h(self.batch_size)
        n = len(h)
        s = np.array([h[i][0] for i in range(n)])
        a = np.array([h[i][1] for i in range(n)])
        r = np.array([h[i][2] for i in range(n)])
        s2= np.array([h[i][3] for i in range(n)])
        done = np.array([h[i][4] for i in range(n)])

        a_predict = self.A_network.choose_action_with_target_net(s2)
        r_predict = self.C_network.predict_reward_with_target_net(s2, a_predict)
        y = r.reshape(-1) + r_predict.reshape(-1) * (1 - done) * self.gamma
        y = y.reshape(-1,1)
        
        a_source = self.A_network.choose_action_with_source_net(s)
        a_source_gradient = self.C_network.get_gradient_of_a_with_source_net(s,a_source)
        self.C_network.learn_with_source_net(s,a,y)
        self.A_network.learn_with_source_net(s, a_source_gradient.reshape(-1,self.action_size))

        self.C_network.learn_with_target_net()
        self.A_network.learn_with_target_net()