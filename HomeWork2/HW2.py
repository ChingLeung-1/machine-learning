import numpy as np
import pandas as pd
import os
from math import floor, log
from numpy.linalg import inv

TrainDataDirectory = 'J:/PythonWorkSpace/HomeWork2/data/train.csv'
TestDataDirectory = 'J:/PythonWorkSpace/HomeWork2/data/test.csv'
AnserDirectory = 'J:/PythonWorkSpace/HomeWork2/data/correct_answer.csv'
OutputDirectory = 'J:/PythonWorkSpace/HomeWork2/outPut/'


# 清洗带问号的数据
def washData(dir1, dir2='nothing'):
    rawData = pd.read_csv(dir1)
    if dir2 != 'nothing':  # 表示是trainData含有label
        df_ans = pd.read_csv(dir2)
        rawData = pd.concat([rawData, df_ans['label']], axis=1)  # 注意训练集里面列名是'income', 这里是'label'
        rawData.rename(columns={'label': 'income'}, inplace=True)  # label -> income
    rawData = rawData.replace(' ?', np.nan)  # 将数据中存在'?'的行用NAN替代
    rawData = rawData.dropna()  # 将含有NAN的行删除
    return rawData


#  处理X数据
def dataProcess_X(rawData):
    # sex 只有两个属性 先drop之后处理
    if "income" in rawData.columns:  # if in 是训练集
        Data = rawData.drop(["sex", 'income'], axis=1)
    else:  # 是测试集
        Data = rawData.drop(["sex"], axis=1)
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"]  # 读取非数字的column
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn]  # 数字的column

    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjedtColumn]

    # 插入性别到非对象数组的第0位
    NonObjectData.insert(0, 'sex', (rawData['sex'] == ' Female').astype(np.int))
    # 对ObjectData进行编码
    ObjectData = pd.get_dummies(ObjectData)
    # print('编码后：', ObjectData)
    # 纵轴拼接
    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    # print('列名:', Data.columns)  # 原本数字的在前，字符的在后，sex是第一个
    # 转化成64位
    Data_x = Data.astype('int64')

    # normalize
    Data_x = (Data_x + Data_x.mean()) / Data_x.std()
    return Data_x


# 处理y数据
def dataProcess_Y(rawData):
    Data_y = (rawData['income'] == ' >50K').astype(np.int64)
    return Data_y


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, (1 - 1e-8))


# 打乱数据顺序
def _shuffle(X, Y):
    randomize = np.arange(X.shape[0])  # [0-32561)
    np.random.shuffle(randomize)  # 洗牌，打乱顺序
    return (X[randomize], Y[randomize])


# 分离TrainingData和validData(cross validation 交叉验证)
def split_valid_set(X, Y, percentage):
    all_size = X.shape[0]
    # 向下取整分离validSet
    valid_set = int(floor(all_size * percentage))  # 3016
    X, Y = _shuffle(X, Y)
    X_valid, Y_valid = X[:valid_set], Y[:valid_set]
    X_train, Y_train = X[valid_set:], Y[valid_set:]
    return X_train, Y_train, X_valid, Y_valid


# valid(验证该数据从当前guassian sample出来的概率大小Most Likelyhood)
def valid(X, Y, mu1, mu2, shared_sigma, N1, N2):
    # 矩阵求逆
    sigma_inv = inv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inv)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(float(N1) / N2)
    # log下什么都不带默认自然对数
    x_t = X.T
    z = np.dot(w, x_t) + b
    # 将分布控制在0-1之间
    y = sigmoid(z)
    # 四舍五入，不是0就是1
    y_ = np.around(y)
    # squeeze()将维度里面为1的值维度去掉。Y(3256,1) y_(3256,)
    result = (np.squeeze(Y) == y_)  # result(3256,) [true or false]

    # 对后半部分的数据进行验证输出平均值
    print("Valid acc = %f" % (float((result.sum())) / result.shape[0]))
    return


# train训练数据
def train(X_train, Y_train):
    train_data_size = X_train.shape[0]
    mu1 = np.zeros((X_train.shape[1]),)
    mu2 = np.zeros((X_train.shape[1]),)
    # mu最优既是数据和再求平均
    count1 = 0
    count2 = 0
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            count1 += 1
        else:
            mu2 += X_train[i]
            count2 += 1
    # 求均值
    mu1 /= count1
    mu2 /= count2

    # 定义sigma(协方差矩阵)
    sigma1 = np.zeros((X_train.shape[1], X_train.shape[1]))
    sigma2 = np.zeros((X_train.shape[1], X_train.shape[1]))

    scount1 = 0
    scount2 = 0

    for i in range(train_data_size):
        if Y_train[i] == 1:
            # transpose()是转置函数  可以用X.T替代??
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])
            scount1 += 1
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [X_train[i] - mu2])
            scount2 += 1
    # 分别求出 sigma1，2
    sigma1 /= scount1
    sigma2 /= scount2

    # 求出共用的sigma
    shared_sigma = (float(scount1) / train_data_size) * sigma1 + (float(scount2) / train_data_size) * sigma2  # 分布∑
    # 各class的样本数量用count1 count2也行？
    N1 = scount1
    N2 = scount2

    return mu1, mu2, shared_sigma, N1, N2


if __name__ == "__main__":

    train_rawData = washData(TrainDataDirectory)
    test_rawData = washData(TestDataDirectory, AnserDirectory)
    # 先得到训练数据
    x_train = dataProcess_X(train_rawData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    y_train = dataProcess_Y(train_rawData).values
    x_test = dataProcess_X(test_rawData).values
    y_ans = test_rawData['income'].values
    # 验证的集合占1成
    vaild_set_percetage = 0.1
    # 返回的是打乱了、分割了的数据
    X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y_train, vaild_set_percetage)
    # 初次训练
    mu1, mu2, shared_sigma, N1, N2 = train(X_train, Y_train)
    # 验证训练结果
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)
    # 开始对整个训练集训练
    mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)
    for i in 20:

    sigma_inv = inv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inv)
    X_t = x_test.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(
        float(N1) / N2)
    a = np.dot(w, X_t) + b
    y = sigmoid(a)
    y_ = np.around(y).astype(np.int)
    # df = pd.DataFrame({"id": np.arange(1, 15061), "label": y_})
    result = (np.squeeze(y_ans) == y_)
    print('Test acc = %f' % (float(result.sum()) / result.shape[0]))
    # df = pd.DataFrame({"id": np.arange(1, 15061), "label": y_})
    # if not os.path.exists(OutputDirectory):
    #     os.mkdir(OutputDirectory)
    # df.to_csv(os.path.join(OutputDirectory + 'gd_output.csv'), sep='\t', index=False)
