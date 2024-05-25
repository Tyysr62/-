import torch
import torchvision
from IPython import display
from d2l import torch as d2l
from torchvision import transforms
from torch.utils import data

def get_dataloader_workers(): #@save
    """使用6个进程来读取数据"""
    return 6    

def load_data_fashion_mnist(batch_size, resize = None):
    trans = [transforms.ToTensor()]
    if (resize):
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=get_dataloader_workers()))

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784 # 每个样本都是28* 28的图像，将这个展平，顾看做784长的向量
num_outputs = 10 # 输出有10个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #每个个像素点，在某个分类上的w_{i,j}
b = torch.zeros(num_outputs, requires_grad=True) #每个分类的偏移 

X = torch.tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
#print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))
#print(X.sum(0, keepdim=False), X.sum(1, keepdim=False))
#print(X.sum(0, keepdim=True).shape, X.sum(0, keepdim=False).shape, X.sum(1, keepdim=True).shape,X.sum(1, keepdim=False).shape)

def softmax(X):
    X_exp = torch.exp(X)
    temp = X_exp.sum(1,keepdim=True)
    return X_exp / temp

#X = torch.normal(0,1,(2,5)) # 此处只是一个验证，和前文无关，相对于有两个输入，分为5个种类
#X_prob = softmax(X) # softmax之后可以发现，所有的值都为正值，对于每一个输入，不同种类的概率只和为1
#print(X_prob,X_prob.sum(1))

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) # W是[784 * 10] X 为[batch_size,[picture]] ---> [256，1，28,28] 要reshape为[256  ,784]才可以做矩阵乘法

# 此处为示例，表示两个样本在三个类别的预测概率
y = torch.tensor([0, 2]) #此处为索引,
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 此处为概率结果
#print(y_hat[[0, 1], y]) #这好像是特性？？？ 可以理解为 print(y_hat[0][0], y_hat[1][2])对于第i个数，取y_hat[i][y[i]]


'''
1. torch.log 默认以e为底
2. 交叉熵的正常公式为−\sum_{i=1}^N y_i \log(\hat y_i)
    在这里的计算中,y_i被隐含了。 在这里的是one-hot编码，此时输入的y为一个(batch_size,) 的张量，
    且只在特性类型出为1 例如[0,0,1,0,...0]，然后直接根据这个进行索引，得出来的就是交叉熵
'''
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y]) #相当于 \sum y_i

#print(cross_entropy(y_hat, y))


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: #检查 y_hat 是否是一个多维张量，且第二个维度的大小大于1。相当于检测y_hat是不是一个概率分布，其中每一行包含了对应各个类别的概率。
        y_hat = y_hat.argmax(axis=1) #获取这个样本最大可能的结果
    #print(y_hat.type,y.type)
    #cmp = y_hat == y
    cmp = y_hat.type(y.dtype) == y # 当前样例下不转其实也可以，应该是为了增强代码的健壮性，将y_hat 的type 显式的转为y的type (由于等式运算符“==”对数据类型很敏感,因此我们将y_hat的数据类型转换为与y的数据类型一致。)
    return float(cmp.type(y.dtype).sum()) #计算 y_hat 预测和 y中结果一样的数量和

#print(accuracy(y_hat, y) / len(y))

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module): # if true 说明是一个pytorch 模型
        net.eval()# 将模型设置为评估模式，模型不会更新梯度，可以提高计算效率。
    metric = Accumulator(2) # 正确预测数、预测总数 
    with torch.no_grad():
        for X, y in data_iter:
            # 1. net(X) 算出\hat y,softmax 转换成概率
            # 2. accuracy 算出最大可能的概率，算出当前batch的预测正确的样本数，add[正确数，样本数]
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n): # 接受一个参数 n，表示要累加的变量的数量，每个初始化为[0,0]
        self.data = [0.0] * n
    def add(self, *args): # 可以接受任意个数的参数，通过zip函数将self.data与args每个一一对应，然后相加
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):# 重置累加器
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx): #索引
        return self.data[idx]
'''
params :
    net : 计算softmax
    train_iter: 训练数据集合
    loss : 使用的是什么loss,这里就是交叉熵增
    updater: 使用的是什么优化器， 这里是随机梯度下降法
'''
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module): # 将模型设置为训练模式
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)# 计算\hat y
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad() # 梯度初始化为0
            l.mean().backward() # 计算梯度
            updater.step() #更新梯度
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]# matric：[loss和，预测正确的个数，总的预测个数]

class Animator: #这段不重要，有这个工具就好了
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                ylim=None, xscale='linear', yscale='linear',
                fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                figsize=(3.5, 2.5)):
    # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        d2l.plt.draw()
        d2l.plt.pause(0.001)
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) # 计算训练损失和训练精度
        test_acc = evaluate_accuracy(net, test_iter) #计算精度
        animator.add(epoch + 1, train_metrics + (test_acc,))# UI显式
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=6): #@save
    """预测标签(定义见第3章)"""
    for X, y in test_iter:
            break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
predict_ch3(net, test_iter)

# practise 1
# 按照提示应该是会溢出 

# practise 2
# 对数定义域为(0,+ \infty) 如果，接近0 也有可能溢出

# parctise 3
# 设置边界？？？

# practise 4
# 可能不行，可能不只有一种病？ 给出所有可能吧

# practise 5
# 计算量太大？ 我看有的答案说单词之间概率差别不大，但是exp函数会放大概率，感觉应该还是能有结果？