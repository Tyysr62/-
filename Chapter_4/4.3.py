import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

#practise 1
'''
根据沐神QA
拿到一个MLP 比如有128个数据,分为2类,则向量机个数为(2,128)个
1. 先试试线性（没有隐藏层）
2. 加一个隐藏层 128 -> 16/32/64 -> 2,如果16和128效果都不行,16太简单,128太复杂,32,64还可以。到步骤3
3. 加两个隐藏层 128 -> 64 -> 32/16/8 -> 2
多试几次，从简单到复杂，老中医把脉就完事了
在相同向量机个数的情况下，我们更倾向与使用更深的，而不是更宽的神经网络；。
激活函数只提供非线性，对于训练的结果影响不大，用ReLu就好了，比较简单
'''

