#coding=utf-8
import numpy as np
from random import shuffle
from numpy.linalg import inv
from math import floor, log
import os
import argparse
import pandas as pd



output_dir = "J:/PythonWorkSpace/HomeWork2/outPut/"

def dataProcess_X(rawData):

    # sex 只有两个属性 先drop之后处理
    if "income" in rawData.columns:  # if in 是训练集
        Data = rawData.drop(["sex", 'income'], axis=1)
    else:  # 是测试集
        Data = rawData.drop(["sex"], axis=1)
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn] #数字的column

    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjedtColumn]
    #insert set into nonobject data with male = 0 and female = 1
    NonObjectData.insert(0 ,"sex", (rawData["sex"] == " Female").astype(np.int))
    #set every element in object rows as an attribute
    # print('编码前：', ObjectData) -------------------
    ObjectData = pd.get_dummies(ObjectData)
    # print('编码后：', ObjectData) -------------------

    Data = pd.concat([NonObjectData, ObjectData], axis=1)  # 列相连接、并列
    # print('列名:', Data.columns)  # 原本数字的在前，字符的在后，sex是第一个
    Data_x = Data.astype("int64")
    # Data_y = (rawData["income"] == " <=50K").astype(np.int)

    #normalize
    # pandas.std() 计算的是样本标准偏差，默认ddof = 1。如果我们知道所有的分数，那么我们就有了总体
    # ——因此，要使用 pandas 进行归一化处理，我们需要将“ddof”设置为 0。
    Data_x = (Data_x - Data_x.mean()) / Data_x.std()  # pandas.mean()求每一列自己的平均值

    ## 保存数字型数据，通过分析此文件，发现它将原来列属性中不同的值分成了新的列。所以文件的列数激增。
    # if "income" in rawData.columns:
    #     Data_x.to_csv('F:/machine/HW2/dta/train_num_data.csv')

    # 疑惑，目前没有进行数据清洗，即数据不全处。
    return Data_x

def dataProcess_Y(rawData):
    df_y = rawData['income']  # 太帅了这个用法， 并且使用的时候我们可以不转换为数组
    Data_y = pd.DataFrame((df_y==' >50K').astype("int64"), columns=["income"])
    return Data_y


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))  # 整体的函数
    return np.clip(res, 1e-8, (1-(1e-8)))  # （输入的数组，限定的最小值，限定的最大值）

def _shuffle(X, Y):                                 #X and Y are np.array
    randomize = np.arange(X.shape[0])  # [0-32561)
    np.random.shuffle(randomize)  # 洗牌，打乱顺序
    return (X[randomize], Y[randomize])

def split_valid_set(X, Y, percentage):
    all_size = X.shape[0]  # 32561
    valid_size = int(floor(all_size * percentage))  # 3256

    X, Y = _shuffle(X, Y)  # 将数据打乱
    # 将数据分成 percentage: 1-percentage两部分
    X_valid, Y_valid = X[ : valid_size], Y[ : valid_size]
    X_train, Y_train = X[valid_size:], Y[valid_size:]

    return X_train, Y_train, X_valid, Y_valid

def valid(X, Y, mu1, mu2, shared_sigma, N1, N2):
    sigma_inv = inv(shared_sigma)  # 矩阵求逆
    w = np.dot((mu1-mu2), sigma_inv)
    X_t = X.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(float(N1)/N2)
    a = np.dot(w,X_t) + b  # 唉，弄了半天，代码没有问题，只不过w在PPT上用的是wT这个名字，但是意义是一样的。
    y = sigmoid(a)  # a就是线性里面的y了，在逻辑回归里面只不过套了哥函数，将其分布改成0-1之间
    y_ = np.around(y)  # 四舍五入的值，即二分类
    # squeeze()将维度里面为1的值维度去掉。Y(3256,1) y_(3256,)
    result = (np.squeeze(Y) == y_)  # result(3256,) [true or false]

    # 我训练集的前半部分得出的函数，对后半部分的测试成功率
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return

def train(X_train, Y_train):
    # vaild_set_percetange = 0.1
    # X_train, Y_train, X_valid, Y_valid = split_valid_set(X, Y, vaild_set_percetange)

    #Gussian distribution parameters
    train_data_size = X_train.shape[0]

    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((X_train.shape[1],))
    mu2 = np.zeros((X_train.shape[1],))
    for i in range(train_data_size):
        if Y_train[i] == 1:     # >50k
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1  # 均值U
    mu2 /= cnt2

    sigma1 = np.zeros((X_train.shape[1], X_train.shape[1]))  # （106，106）
    sigma2 = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            # sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])  # 分布∑1  # 公式有误？？
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])  # 分布∑1  # 公式有误？？

        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [X_train[i] - mu2])  # 分布∑2

    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2  # 分布∑

    N1 = cnt1
    N2 = cnt2
    return mu1, mu2, shared_sigma, N1, N2  # 现在将公式的参数全部求出了


if __name__ == "__main__":
    trainData = pd.read_csv("J:/PythonWorkSpace/HomeWork2/data/train.csv")  # 第一行会作为列名 （32561，15）
    testData = pd.read_csv("J:/PythonWorkSpace/HomeWork2/data/test.csv")  # （16281，14）没有数据
    ans = pd.read_csv("J:/PythonWorkSpace/HomeWork2/data/correct_answer.csv")  # （16281， 2） 2 = id + label

    #here is one more attribute in trainData
    # 删除训练集中 有['native_country_ Holand-Netherlands']的那一列， 因为测试集里面无此国家项，即无此列
    x_train = dataProcess_X(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values   # （32561，107-1）
    x_test = dataProcess_X(testData).values  # （16281，106）
    y_train = dataProcess_Y(trainData).values  # （32561，1）
    y_ans = ans['label'].values  # （16281,）if 达到50K then 1 else 0 answer for test

    vaild_set_percetage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y_train, vaild_set_percetage) # 返回的是打乱了、分割了的数据
    mu1, mu2, shared_sigma, N1, N2 = train(X_train, Y_train)

    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)

    mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)  # 开始对整个训练集训练
    sigma_inv = inv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inv)
    X_t = x_test.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(
        float(N1) / N2)
    a = np.dot(w, X_t) + b
    y = sigmoid(a)
    y_ = np.around(y).astype(np.int)
    print(len(y_))
    df = pd.DataFrame({"id" : np.arange(1,16282), "label": y_})
    result = (np.squeeze(y_ans) == y_)
    print('Test acc = %f' % (float(result.sum()) / result.shape[0]))
    df = pd.DataFrame({"id": np.arange(1, 16282), "label": y_})
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir+'gd_output.csv'), sep='\t', index=False)









