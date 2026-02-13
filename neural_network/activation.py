import numpy as np


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, dE):
        return dE * (self.x > 0).astype(np.float32)


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, dE):
        return dE * (self.out * (1.0 - self.out))


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dE):
        return dE * (1.0 - self.out ** 2)


class Softmax:
    def __init__(self):
        self.out = None

    def forward(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    @staticmethod
    def backward(dE):
        return dE
