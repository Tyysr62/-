import torch
from torch.distributions import multinomial
from d2l import torch as d2l

# # multinomial.Multinomial() 是 PyTorch 中的一个类，用于模拟多项分布。
# fair_probs = torch.ones([6]) / 6 # 概率向量，每个概率\frac{1}{6} 
# #print (multinomial.Multinomial(1, fair_probs).sample())#开始抽样1次，每次运行结果都不同
# #print(multinomial.Multinomial(10, fair_probs).sample())#开始抽样10次

# counts = multinomial.Multinomial(10000, fair_probs).sample()
# print(counts / 10000)

# practise 1
fair_probs = torch.ones([500]) / 500
print (multinomial.Multinomial(10, fair_probs).sample())