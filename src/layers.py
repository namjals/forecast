import numpy as np
from functions import mean_squared_error


class Operation:
    def __init__(self, m, b):
        self.params = [m, b]  # 학습 대상
        self.grads = [np.zeros_like(m), np.zeros_like(b)]  # 역전파 미분값

    def forward(self, x):
        m, b = self.params
        self.m = m
        self.b = b
        self.x = x
        out = m * x + b
        return out

    def backward(self, dout):
        dm = -self.x * (dout - (self.m * self.x + self.b))
        db = -(dout - (self.m * self.x + self.b))

        dm *= 2.0 / float(dm.shape[0])
        db *= 2.0 / float(db.shape[0])
        self.grads[0][...] = np.average(dm)
        self.grads[1][...] = np.average(db)


class MSE:
    '''
    Mean Squared Error
    '''

    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # mx + b 값
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.y = x
        self.t = t
        loss = mean_squared_error(x, t)
        return loss

    def backward(self, dout=1):
        return self.t