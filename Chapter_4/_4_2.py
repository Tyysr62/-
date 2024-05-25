import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256 # 每个图像 28 * 28 （728））个像素，分成10个类别，拥有256个隐藏单元

# 隐藏层
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01) # 随机 W有784行，256列 ，需要梯度 784输入到隐藏层有256个参数
# W1 = nn.Parameter(torch.ones(
#     num_inputs, num_hiddens, requires_grad=True) * 0.01) # 随机 W有784行，256列 ，需要梯度
# W1 = nn.Parameter(torch.zeros(
#     num_inputs, num_hiddens, requires_grad=True) * 0.01) # 随机 W有784行，256列 ，需要梯度
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) # 随机 W有784行，256列 ，需要梯度

# 输出层
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01) #隐藏层 256 到输出层 10
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

#定义模型
def net(X):
    X = X.reshape((-1, num_inputs)) # 图片拉成矩阵
    H = relu(torch.matmul(X,W1) + b1)
    return (H @ W2 + b2) # @ = torch.matmul

loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.5
#updater = torch.optim.SGD(params, lr=lr)
#d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
#精度其实并没有提示很多，这里loss可能不显示，应该是loss太大了

# 这里通过改沐神中train_ch3函数，记录了epoch中测试精度最高的结果best_test_acc
#practise 1
#   num_hidden       256        20      700        啊？ 啥情况，难道要用final ???
#   best_test_acc   0.8641      0.8065  0.8394

#practise 2
# num_inputs, num_outputs, num_hiddens_1, num_hiddens_2 = 784, 10, 256, 32 # 每个图像 28 * 28 （728））个像素，分成10个类别，拥有256个隐藏单元
# W1 = nn.Parameter(torch.randn(
#     num_inputs, num_hiddens_1, requires_grad=True) * 0.01) 
# b1 = nn.Parameter(torch.zeros(num_hiddens_1, requires_grad=True)) 

# W2 = nn.Parameter(torch.randn(
#     num_hiddens_1, num_hiddens_2, requires_grad=True) * 0.01)
# b2 = nn.Parameter(torch.zeros(num_hiddens_2, requires_grad=True))

# W3 = nn.Parameter(torch.randn(
#     num_hiddens_2, num_outputs, requires_grad=True) * 0.01)
# b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# params = [W1, b1, W2, b2, W3, b3]
# updater = torch.optim.SGD(params, lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

#def net(X):
#   X = X.reshape((-1, num_inputs)) # 图片拉成矩阵
#   H = relu(torch.matmul(X,W1) + b1)
#   H = relu(H @ W2 + b2)
#   return (H @ W3 + b3) # @ = torch.matmul
# 0.8641 ----- > 0.846 下降了

#practise 3
#   lr              0.1        0.02      0.5        啊？ 啥情况，难道要用final ???
#   best_test_acc   0.8641     0.8082   0.8577

#practise 4
# 慢慢调参吧QAQ

#practise 5
# 太麻烦了，不然我为什么不写第四题 -> -> 

#practise 6
# 从成本最小的参数开始调，每次确定一个最优的参数