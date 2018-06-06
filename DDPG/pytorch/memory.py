import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def size(self):
        return len(self.buffer)

    def append(self, obj):
        if self.size() < self.buffer_size:
            self.buffer.append(obj)
        else:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size

    def sample_batch(self, batch_size):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        item_count = len(batch[0])
        memory = []
        for i in range(item_count):
            tmp = np.stack((item[i] for item in batch), axis=0)
            if len(tmp.shape) == 1:
                tmp.shape += (1,)
            memory.append(tmp)

        [state_batch, action_batch, reward_batch, next_state_batch, terminal_batch] = memory
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch


