# 矢量化加速(向量化加速 vectorrized)
# 矢量化加速是一种优化计算的方法，特别适用于处理大规模数据和高维度数组。
# 向量化计算：在编程中，向量化计算是将多次循环计算转换为一次计算的过程。
#   它利用CPU的SIMD（单指令，多数据）指令集，可以同时处理多份数据。通过向量化，
#   我们可以将一次循环操作应用于整个数组，从而提高计算效率。
# 优势：向量化计算可以大幅度提高计算速度。通过使用SIMD指令集，我们可以将循环操作转化为一次计算，
#   从而减少了循环开销和内存访问次数
# 原因：Python是解释语言，这意味着你的指令进行分析，并在每次执行解释。
#   由于它们不是静态类型的，因此循环必须在每次迭代时评估操作数的类型，这导致计算开销。
#   向量化可以使一条指令并行地对多个操作数执行相同的操作（SIMD（单指令，多数据）操作）。
#   例如，要将大小为N的向量乘以标量，让我们调用M可以同时操作的操作数大小。如果是这样，
#   那么它需要执行的指令数大约为N / M，如果采用循环方式，则必须执行N次操作。

import math
import time
import numpy as np
import torch
from d2l import torch as d2l

n = 100000000
a = torch.ones([n])
b = torch.ones([n])

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器"""
        self.tik = time.time()
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec') # 0.06283 sec

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec') # 0.00051 sec
# 容量大小          10000       1000000 
# for循环用时       0.05262     5.25166 线性？
# vectorrized      0.00051     0.00100 指数？

def gaussian(x,mu,sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)