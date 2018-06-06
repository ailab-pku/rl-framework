from Agent import Agent
from Memory_buffer import Memory_buffer
from DDPG import DDPG_model
import tensorflow as tf
import numpy as np
import gym

BATCH_SIZE = 64
LEARNING_START_POINT = 5 * BATCH_SIZE
MAX_EXPLORE_EPS = 100

def turn(s):
    s = np.reshape(s,(-1,))
    s.astype(np.float32)
    return s

if __name__ == '__main__':
    RENDER = 0
    model = DDPG_model(3,2,batch_size = BATCH_SIZE)
    agent = Agent(model, Memory_buffer())
    env = gym.make('Pendulum-v0')  
    #env = env.unwrapped     # 取消限制
    env.seed(1)
    for i_episode in range(3000):

        state = env.reset()
        state = turn(state)
        agent.init()
        sum_r = 0
        time = 0
        sum = 0
        while True:

            if RENDER: env.render()
            time += 1
            noise = 0
            sum += np.random.random_sample() * 0.02 - 0.01
            if time <= MAX_EXPLORE_EPS:
                noise = sum * time / MAX_EXPLORE_EPS
            if i_episode >= 5:
                noise = 0
            action = agent.choose_action(state, noise)   # 返回的action形如[[1,2,3,...(action_size)个]],每一维取值-1~1
            action = action[0]
            action_ = action * 2
            next_state, reward, done, info = env.step([action_[0]+action[1]])
            sum_r += reward
            
            next_state = turn(next_state)
                
            agent.store_transition(state, action, reward, next_state, done)
            if time % 10 == 0:
                agent.learn_when_enough(LEARNING_START_POINT)
            state = next_state

            if done:
                break

        print('return of ' + str(i_episode)+' is '+ str(sum_r))