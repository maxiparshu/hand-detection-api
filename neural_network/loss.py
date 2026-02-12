import numpy as np


class CrossEntropy:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.y_true = None
        self.y_hat = None
        self.loss = None

    def forward(self, y_true, y_hat):
        self.y_true = y_true
        self.y_hat = np.clip(y_hat, self.epsilon, 1.0 - self.epsilon)

        self.loss = -np.sum(self.y_true * np.log(self.y_hat))
        return self.loss

    def backward(self):
        return self.y_hat - self.y_true

    def forward_batch(self, y_true, y_hat):
        self.y_true = y_true
        self.y_hat = np.clip(y_hat, self.epsilon, 1.0 - self.epsilon)

        batch_loss = -np.sum(self.y_true * np.log(self.y_hat), axis=1)
        self.loss = np.mean(batch_loss)
        return self.loss

    def backward_batch(self):
        return self.y_hat - self.y_true
