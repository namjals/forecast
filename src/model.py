import numpy as np
from layers import Operation, MSE

class LinearRegression:
    def __init__(self):
        # 기울기와 편향 초기화
        m = 0.01 * np.random.randn(1, 1)
        b = 0.01 * np.random.randn(1, 1)

        # 계층 생성
        self.layers = [
            Operation(m, b)
        ]

        self.loss_layer = MSE()

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.params
        return params