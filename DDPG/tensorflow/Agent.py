import numpy as np

class Agent:

    def __init__(self, model, memory_buffer):

        self.model = model
        self.memory_buffer = memory_buffer
        pass

    def init(self):
        pass
    
    def choose_action(self, present_state, noise = 0, p = 1):
        
        return self.model.choose_action(present_state) * p + noise * (1 - p)

    def store_transition(self, s, a, r, s2, done):

        a = np.reshape(a,(-1))
        self.memory_buffer.store_transition([s, a, r, s2, done])

    def learn(self):

        self.model.learn(self.memory_buffer)

    def learn_when_enough(self, threshold):

        if self.memory_buffer.len() >= threshold:
            self.model.learn(self.memory_buffer)


