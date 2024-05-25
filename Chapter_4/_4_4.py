import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import ..Chapter_4._

# 用 y = 5 + 1.2 * x - 3.4 * x * x / 2! + 5.6 * x * x * x / 3! + noise    noise 服从 N （0,0.1^2）的高斯分布
max_degree = 20 # 多项式阶数，越大越容易过拟合
n_train, n_test = 100,100 # 训练和测试数据集大小
true_w = np.zeros(max_degree) # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])  # 更高的阶全是0

features = np.random.normal(size=(n_train+n_test, 1)) # 生成一个长度为n_train+n_test tensor，满足的标准正态分布
np.random.shuffle(features) # 打乱
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) #poly_features[i][j] 随机的特征i 的 j次方，这里应该是为了方便索引？
#print(features.shape, poly_features.shape, np.arange(max_degree).reshape(1, -1).shape)
for i in range (max_degree):
    poly_features[: i] /= math.gamma(i + 1) # 计算多项式的 x^n / n! 这样后面只需要乘以 W_i 就能完成计算
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
                                        torch.float32) for x in [true_w, features, poly_features, labels]]

def evaluate_loss(net = d2l.data., data_iter, loss):
    metric = d2l.Accumulator(2)
    for X,y in data_iter:
        out