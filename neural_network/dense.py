import numpy as np


class Dense:
    def __init__(self, input_neurons, output_neurons, reg_lambda=0.0, momentum=0.9):
        self.dx = None
        self.x = None
        self.W = np.random.randn(input_neurons, output_neurons) * np.sqrt(2.0 / input_neurons)
        self.b = np.zeros(output_neurons)

        self.reg_lambda = reg_lambda
        self.momentum = momentum

        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.final_dW = np.zeros_like(self.W)
        self.final_db = np.zeros_like(self.b)

    def backward(self, dE, learning_rate=0.01, mini_batch=False, update=True, len_mini_batch=None):
        d_w = self.x.T @ dE
        db = np.sum(dE, axis=0)
        self.dx = dE @ self.W.T

        if self.reg_lambda != 0:
            d_w += self.reg_lambda * self.W

        if mini_batch:
            self.final_dW += d_w
            self.final_db += db

            if update and len_mini_batch:
                avg_d_w = self.final_dW / len_mini_batch
                avg_db = self.final_db / len_mini_batch

                self.vW = self.momentum * self.vW - learning_rate * avg_d_w
                self.vb = self.momentum * self.vb - learning_rate * avg_db

                self.W += self.vW
                self.b += self.vb

                self.final_dW.fill(0)
                self.final_db.fill(0)

        return self.dx

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def get_weight(self):
        return self.W, self.b

    def set_weight(self, W, b):
        self.W = W
        self.b = b


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, x, train=True):
        if not train:
            self.mask = np.ones_like(x)
            return x

        self.mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dz):
        return dz * self.mask

    def get_p(self):
        return self.p

    def set_p(self, new_p):
        self.p = new_p
