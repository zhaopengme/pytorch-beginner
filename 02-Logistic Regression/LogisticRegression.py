import torch

import numpy as np

d = np.array([
    [1, 34, 88,333],
    [6, 42, 5,4],
    [253, 34, 1,2]
])

x = torch.from_numpy(d)
a, b = torch.max(x, 1);

print(a)
print(b)# b代表从哪个位子的取值
#
# print(torch.max(x,1))# max(data,dim) dim 取值是从-2到1,取某个维度的最大值
# print(torch.max(x,0))
