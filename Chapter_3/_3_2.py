import random
import torch
from d2l import torch as d2l

# y = Wx + b + noise 合成数据
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
#print('features:', features[0],'\nlabel:', labels[0])

def data_iter(batch_size, features, labels):
    num_examples = len(features) # 样本数量
    indices = list(range(num_examples)) #生产一个迭代器，包含[0， num_examples]的数据
    random.shuffle(indices)# 随机打乱迭代器 ------>   这里体现了随机梯度下降中的随机概念，既采样的随机的
    for i in range (0, num_examples, batch_size):# for(i = 0 ; i < num_examples; i += batch_size)
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]# generator 类似于直接赋值，但不需要反复调用内存

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 初始化参数模型 weights bias
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
# practise 1 
#w = torch.zeros(size =(2,1),requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 参数 \hat y = Weights * X + bias
def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y): #@save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad(): # 在使用过程中不需要梯度
        for param in params: # 依次更新参数 w ,b 
            param -= lr * param.grad / batch_size # w_i = w_{i-1} - 学习率 * 梯度方向 \ batch_size 损失函数中没有求均值，在这里求一样的。感觉这样可读性变差了 = =
            param.grad.zero_() # 梯度设置为 0 ，使得两次之间计算无关

learnRate = 0.03 # 学习率，步长
num_epochs = 10 # 迭代次数
net = linreg #线性回归
loss = squared_loss # L2 loss

for epoch in range(num_epochs): 
    for X, y in data_iter(batch_size, features, labels): # 每次拿出 batch_size 大小的X 和 Y 
        l = loss(net(X, w, b), y) # 计算 当前的残差 
        #print(l.shape)
        # X和y的小批量损失
        # 因为l形状是(batch_size,1),而不是一个标量。l中的所有元素被加到一起,
        # 并以此计算关于[w,b]的梯度
        l.sum().backward() # 计算当前batch 的 梯度
        sgd([w, b], learnRate, batch_size)# 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

# practise 1
'''
    也能跑， 但效果不好  ---> https://zhuanlan.zhihu.com/p/75879624
'''
# practise2 
# 过

# practise 3 
#  啊？

 