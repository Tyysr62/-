import numpy as np
import torch
from torch.utils import data # https://pytorch.org/docs/stable/data.html
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def huberLoss(y_hat, y, sigma = 0.005):
    error = abs(y_hat.detach().numpy() - y.detach().numpy())
    return torch.tensor(np.where(error < sigma, error - sigma / 2, error ** 2 / (2 * sigma)) , requires_grad= True).mean()

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

"""构造一个PyTorch数据迭代器"""
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays) # 在Python中，*号表示解包操作，它可以将一个包含多个元素的元组、列表、集合等数据结构解压为多个独立的元素。
    return data.DataLoader(dataset, batch_size, shuffle=is_train) #shuffle->数据在每个epoch开始时将被打乱。减少模型训练过程中的过拟合

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

loss = huberLoss

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差:', true_b - b)

#practise 1 
#  / n

#practise 2
# def huberLoss(y_hat, y, sigma = 0.005):
#     error = abs(y_hat.detach().numpy() - y.detach().numpy())
#     return torch.tensor(np.where(error < sigma, error - sigma / 2, error ** 2 / (2 * sigma)) , requires_grad= True).mean()
