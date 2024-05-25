import torch
# x = torch.arange(4.0,requires_grad=True)
# # print(x.grad)
# y = 2 * torch.dot(x, x)
# # print(y)
# y.backward() # 计算 y 相对 x的梯度
# x.grad
# # print(x.grad)

# x.grad.zero_()
# y = x.sum()
# y.backward()
# x.grad
# # print(x.grad)

# # 对非标量调用backward需要传入一个gradient参数,该参数指定微分函数关于self的梯度。
# # 本例只想求偏导数的和,所以传递一个1的梯度是合适的
# x.grad.zero_()
# y = x * x
# # 等价于y.backward(torch.ones(len(x)))
# y.sum().backward()
# print(y,y.sum())
# x.grad
# print(x.grad)

# x.grad.zero_()
# y = x * x
# u = y.detach() # detach 方法用于分离张量，u是y的副本，但不在关联计算
# z = u * x
# print(z)

# z.sum().backward()
# print(x.grad)
# #print(u.grad)
# #print(y.grad)

# pratcise 1
# 过

# practise 2
x = torch.arange(6,dtype=float,requires_grad= True).reshape(2,3)
y = torch.pow(x,3)
loss = y.mean()
loss1 = loss.backward()
print(x.grad)
print(loss1)