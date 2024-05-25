import torch

#y = torch.tensor(3.0)
#x = torch.tensor(2.0)

# print(x + y)
# print(x - y)
# print(x * y)
# print(x / y)
# print(x ** y)

#x = torch.arange(4)
#print(x)

# x = torch.arange(36).reshape(6,6)
# print(x ,"\n")
# #print(x[1])
# #print(x.shape)
# print(x,id(x))
# x = x.T
# print(x,id(x))
# x[:] = x.T
# print(x,id(x)) #无法原地？？？

# A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# print(A)
# print(A.shape)
# print(A.sum)

# A_sum_axis0 = A.sum(axis = 0)# 行来降维
# A_sum_axis1 = A.sum(axis = 1)# 列来降维
# print(A_sum_axis0,A_sum_axis0.shape)
# print(A_sum_axis1,A_sum_axis1.shape)

# x = torch.tensor([1,2,3,4], dtype= torch.float32)
# y = torch.ones(4, dtype = torch.float32)
# print(torch.dot(x,y))

# u = torch.tensor([5.0,12.0])
# print(torch.norm(u))# L2 norm
# print(torch.abs(u).sum())

#practise 1
# x = torch.randn(3,4)
# print(x)
# y = x.T
# print(y)
# z = y.T
# print(z)
# print('x == z ? ', {x == z})

#practise 2
# a = torch.randn(3,4)
# b = torch.randn(3,4)
# c = a.T + b.T
# d = (a + b).T
# print(c)
# print(d)
# print('c == d ? ', {c == d})

#practise 3
# a = torch.randn(4,4)
# b = a + a.T
# print('b == b^/top ? ', {b == b.T})

#practise 4
# a = torch.arange(24,dtype=torch.float32).reshape(2,3,4)
# print('len of tensor [2,3,4] is ',len(a))

#practise 5
# YES, asix = 0

#practise 6
a = torch.arange(30, dtype = float).reshape(2,3,5)
print(a.sum(axis= 0).shape)
print(a.sum(axis= 1).shape)
b = a / a.sum(axis = 0)
print(b)
c = a / a.sum(axis = 1)
print(a.sum(axis= 1).shape)

# a = torch.arange(120, dtype = float).reshape(2,3,4,5)
# b = torch.arange(10, dtype = float).reshape(2,1,1,5)
#print((a + b).shape)
# 不得行 a [2, 3, 5] a.axis[0] = [3,5] 广播 OK ，a[2] = [2,5] 广播NG
# 广播机制 1,或者没有就为通配符

#practise 7
# [3,4]   [2,4]   [1,2]

#practise 8
# 到原点的欧式距离 
