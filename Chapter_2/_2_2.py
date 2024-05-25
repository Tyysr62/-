# 2.2 数据处理

#pratice 1
import os
#os.makedirs(os.path.join('..', 'data'), exist_ok=True)
#data_file = os.path.join('..', 'data', 'house_tiny.csv')
#with open(data_file, 'w') as f:
#    f.write('NumRooms,Alley,Price\n') # 列名
#    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
#    f.write('2,NA,106000\n')
#    f.write('4,NA,178100\n')
#    f.write('NA,NA,140000\n')

import pandas as pd

#data = pd.read_csv(data_file)
#print(data)

#inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
#inputs = inputs.fillna(inputs.mean(numeric_only=True))
#print(inputs)

#pratice 
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file1 = os.path.join('..', 'data', 'animal.csv')
with open(data_file1, 'w',encoding='utf8')as f: #open文件名参数不要打引号
    f.write('动物,年龄,特点\n')
    f.write('马,4,跑的快\n')
    f.write('猪,5,喜欢吃\n')
    f.write('羊,NA,NA\n')
    f.write('鸡,NA,会下蛋\n')
    f.write('牛,3,会吃草\n')
    f.write('NA,NA,NA\n')
data = pd.read_csv(data_file1)
#print(data)

#method 1
# def drop_max_col(m):
#     num = m.isna().sum()
#     print(num)
#     num_dict = num.to_dict()
#     print(num_dict)
#     max_key = max(num_dict,key=num_dict.get) #取字典中最大值的键
#     del m[max_key] #删除缺失值最多的列
#     return m

# drop_max_col(data)

#method 2
tmp = data.isna().sum()
data = tmp.drop(columns=tmp.index[tmp.argmax()])
print(data)


#pratice 2
import torch
#X = torch.tensor(data.to_numpy(dtype = 'flaot'))
#print(X)
