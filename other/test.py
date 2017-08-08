import torch
import numpy as np

torch.manual_seed(12345)# 手动设置随机因子,保证随机出来的结果一致
x = torch.rand(3,1)
print(x)

y = torch.rand(3,1)
print(y)
print(x==y)

x = np.array([[3,34,39],[1,345,64],[2,456,41],[56,3453,23]])
print(x)
x = torch.from_numpy(x)
print(x)


y = np.array([[43],[23],[2],[56]])
print(y)
y = torch.from_numpy(y)
print(y)


print(x == y)


print('00000000000000000000000000000')
print(x)

a = torch.range(1, 16)

print(a)
a = a.view(4, -1)
a = a.view(4, 4)

print(a)
