import numpy as np
import random
class Memory_buffer():
    
    def __init__(self, limit = 1000000):
        self.h = []
        self.length = 0
        self.now = 0
        self.limit = limit
        pass

    def store_transition(self, present_h):
        if self.length == self.limit:
            self.h[self.now] = present_h
            self.now += 1
            if self.now == self.limit:
                self.now = 0
        else:
            self.h.append(present_h)
            self.length += 1

    def len(self):
        return self.length

    def choose_n_h(self, n):
        n = min(n, self.length)
        return random.sample(self.h, n)

    def reset(self):
        self.h = []
        self.length = 0




