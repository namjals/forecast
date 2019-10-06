import numpy as np
from util import read_data
from model import LinearRegression
from optimizer import SGD
from functions import least_error_square

#  방법 1 LinearRegression 사용
data = read_data('../data/count_by_date.csv')
x, t = list(map(lambda x: [int(x)], data['INDEX'])), list(map(lambda x: [int(x)], data['CNT']))
x, t = np.array(x), np.array(t)
model = LinearRegression()
optimizer = SGD(lr=0.08)
epoch = 10000

for e in range(epoch):
    loss = model.forward(x, t) # 오차값
    model.backward()
    optimizer.update(model.params, model.grads)
    m, b = model.get_params() # 기울기, 편향
    # print(loss, m[0][0], b[0][0]) #

print(loss, m[0][0], b[0][0])
print(model.predict(np.array([[i] for i in range(31, 61)]))) # INDEX가 31부터 60까지 추론

# 방법 2 최소오차제곱법 사용
data = read_data('../data/count_by_date.csv')
x, t = list(map(lambda x: [int(x), 1], data['INDEX'])), list(map(lambda x: [int(x)], data['CNT']))
x, t = np.array(x), np.array(t)
m, b = least_error_square(x, t)
print(m, b) # 기울기, 편향
forecast_x = np.array([[i] for i in range(31, 61)]) # INDEX가 31부터 60까지 추론
print(m * forecast_x + b)