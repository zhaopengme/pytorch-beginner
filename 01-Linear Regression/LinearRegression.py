import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.autograd.function as F
import torch.autograd.variable as Fariable

import matplotlib

# 设置为 TkAgg 否则在 mac 上面可能不能显示出来
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import numpy as np

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)





# Linear Regression Model
# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()











plt.figure()
plt.plot(x_train.numpy(),y_train.numpy())
plt.show()