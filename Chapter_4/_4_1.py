import matplotlib
import torch
from d2l import torch as d2l

# RELU 
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
#y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# d2l.plt.show()

# y.backward(torch.ones_like(x), retain_graph=True)
# d2l.plot(x.detach(),x.grad,'x','grad of relu', figsize=(5, 2.5))
# d2l.plt.show()

# sigmoid \frac{1} {1 + \exp (-x)}  输入为0时,sigmoid函数的导数达到最大值0.25  
#y = torch.sigmoid(x)
#d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
#d2l.plt.show()

#y.backward(torch.ones_like(x),retain_graph=True)
#d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
#d2l.plt.show()

# tanh \frac {1 - \exp (-2x)} {1 + \exp (-2x)}
#y = torch.tanh(x)
#d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
#d2l.plt.show()

# y.backward(torch.ones_like(x),retain_graph=True) #当输入接近0时,tanh函数的导数接近最大值1
# d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
# d2l.plt.show()

# practise 1
#pReLU max(0,x) + \alpha min(0,x)
# alpha = torch.tensor(0.1)
# y = torch.prelu(x,alpha)
# d2l.plot(x.detach(), y.detach(), 'x', 'pReLu(x)', figsize=(5, 2.5))
# d2l.plt.show()

# y.backward(torch.ones_like(x),retain_graph=True)
# d2l.plot(x.detach(), x.grad, 'x', 'grad of pReLu', figsize=(5, 2.5))
# d2l.plt.show()

# practise 3
# x = torch.arange(-8.0, 8.0, 0.1)
# y_1 = torch.tanh(x) + 1
# y_2 = 2 * torch.sigmoid(2*x)
# d2l.plot(x.detach(), [y_1.detach(), y_2.detach()],"X","Y",legend=['tanh(x)+1','2sigmoid(2x)'], figsize=(5, 2.5))
# d2l.plt.show()


