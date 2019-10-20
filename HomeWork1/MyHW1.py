import csv

import numpy as np

# 数据读取
train_x = []
train_y = []
data = []
text = open('J:/PythonWorkSpace/HomeWork1/data/train.csv', 'r', encoding='big5')
row = csv.reader(text, delimiter=",")
n_row = 0
for r in row:
    if n_row != 0:
        data.append([])
        # 每一列只有第3-27格有值(1天內24小時的數值)
        # 拿出每行3-27列的数据
        for i in range(3, 27):
            if r[i] != "NR":
                data[n_row - 1].append(float(r[i]))
            else:
                data[n_row - 1].append(float(0))
    n_row = n_row + 1
text.close()

"""
一天中，每8个小时为一组trainData
trainData中包涵NO，Nox，PM10，PM2.5
train_x[[[NO],[Nox][PM10],[PM2.5]],[[NO],[Nox][PM10],[PM2.5]]]
train_y[y1,y2]
"""
# 数据处理
# 将数据集拆分为多个数据帧  每一步是19
for i in range(0, len(data), 18):
    y_out = []
    x_out = []
    for j in range(24 - 9):
        NO = data[i + 4][j:j + 9]
        # Nox = data[i + 6][j:j + 9]
        # PM10 = data[i + 8][j:j + 9]
        PM2p5 = data[i + 9][j:j + 9]
        y_out.append(data[i + 9][j + 9])
        x_out.append(NO)
        # x_out.append(Nox)
        # x_out.append(PM10)
        x_out.append(PM2p5)
    # 每行取9列作为一组训练数据
    train_x.append(x_out)
    train_y.append(y_out)

train_y = np.array(train_y)
train_x = np.array(train_x)

# model: y = Wno*X+Wnox*X+Wpm10*X+Wpm2.5*X+bias + λ*regularization
loss = 0
# initial bias
bias = 0
# inital learning rate
l_r = 1
# initial W1
w1 = np.ones(9)
# initial W2
w2 = np.ones(9)

"""# initial W3
w3 = np.ones(5)
# initial W4
w4 = np.ones(5)"""

# regularization
re_rate = 1

# 用于存放偏置值的梯度平方和
bg2_sum = 0
# 用于存放权重的梯度平方和
w1g2_sum = np.zeros(9)
w2g2_sum = np.zeros(9)

# training round
iteration = 1000
# 检查矩阵
print(train_x.shape)
print(train_y.shape)
# iteration
for i in range(iteration):
    b_grad = 0
    w1_gard = np.zeros(9)
    w2_gard = np.zeros(9)
    # n is index of day
    # train_x[n][j]
    for n in range(len(train_x)):
        for j in range(15):
            x1 = 0
            y = 0
            x2 = 1
            x1 += 2
            y += 1
            x2 += 2
            b_grad += (train_y[n][y] - w1.dot(train_x[n][x1]) - w2.dot(train_x[n][x2]) - bias) * (-1)
            for k in range(9):
                w1_gard[k] += ((train_y[n][y] - w1.dot(train_x[n][x1]) - bias) * (-train_x[n][x1][k]))
                w2_gard[k] += ((train_y[n][y] - w2.dot(train_x[n][x2]) - bias) * (-train_x[n][x2][k]))

    # 求平均
    b_grad /= len(train_x)
    w1_gard /= len(train_x)
    w2_gard /= len(train_x)

    # regularization

    for m in range(9):
        w1_gard[m] += re_rate * w1_gard[m]
        w2_gard[m] += re_rate * w2_gard[m]

    # adagrad

    bg2_sum += b_grad ** 2
    w1g2_sum += w1_gard ** 2
    w2g2_sum += w2_gard ** 2

    # update bias&weight
    bias -= l_r / bg2_sum ** 0.5 * b_grad
    w1 -= l_r / w1g2_sum ** 0.5 * w1_gard
    w2 -= l_r / w2g2_sum ** 0.5 * w2_gard

    # 每训练20轮，输出一次在训练集上的损失
    if i % 200 == 0:
        loss = 0
        for n in range(len(train_x)):
            for j in range(15):
                x1 = 0
                y = 0
                x2 = 1
                x1 += 2
                y += 1
                x2 += 2
                loss += (train_y[n][y] - w1.dot(train_x[n][x1]) - w2.dot(train_x[n][x2]) - bias) ** 2
        print('after {} epochs, the loss on train data is:'.format(i), loss / len(train_x))
