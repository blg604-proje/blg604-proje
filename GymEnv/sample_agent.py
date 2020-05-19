import numpy as np

class Agent(object):
    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob, reward, done):
        return np.tanh(np.random.randn(self.dim_action)) # random action