import torch

x = torch.arange(12)

print(x)
print(x.shape)


re_X = x.reshape(3,4)
print(re_X)

zero_X = torch.zeros((2,3,4))
print(zero_X)

one_X = torch.ones((2,3,4))
print(one_X)

randon_X = torch.randn(3,4)
print(randon_X)

num_X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(num_X)

X = torch.arange(12, dtype=torch.float32).reshape(3,4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(X[-1])
print(X[1:3])

Z = torch.zeros_like(Y)
print('id(z):',id(Z))
Z[:] = X + Y
print('id(z):',id(Z))s
Z = X + Y
print('id(z):',id(Z))

print(X)
print(Y)

print('x > y', X > Y)
print('x = y', X == Y)
print('x < y', X < Y)