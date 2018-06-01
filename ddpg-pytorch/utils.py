import numpy as np
import torch


class Noise(object):
    def __init__(self, size):
        self.size = size

    def sample(self):
        return np.random.normal(size=self.size)


class OrnsteinUhlenbeckProcess(object):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def to_numpy(tensor):
    return tensor.to(torch.device("cpu")).detach().numpy()


def to_tensor(ndarray, dtype=torch.float, device=torch.device("cpu"), requires_grad=False):
    return torch.tensor(ndarray, dtype=dtype, device=device, requires_grad=requires_grad)


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)
