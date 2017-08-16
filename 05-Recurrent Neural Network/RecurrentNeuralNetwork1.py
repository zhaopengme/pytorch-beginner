# rnn 回归 拟合
import torch
from  torch.autograd import Variable
from  torch import nn, optim
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

torch.manual_seed(1)  # 手动设置随机因子,保证每次运行,随机数据一样

# 定义超参数
time_step = 10
input_size = 1
learning_rate = 0.01

# 打印一下测试数据
steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)
np_x = np.sin(steps)
np_y = np.cos(steps)

plt.plot(steps, np_y, 'r-', label='target (cos)')
plt.plot(steps, np_x, 'b-', label='input (sin) ')
plt.legend(loc='best')
plt.show()


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,  # 每个时间点都要记录,所有为1
            hidden_size=32,  # run hidden unit 神经元数量,输出 32
            num_layers=1,  # rnn 有几层
            batch_first=True  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.linear = nn.Linear(
            32,  # self.rnn 输出32个神经元
            1  # 返回一个点的值
        )

    def forward(self, x, h_state):# 因为 h_state 是一直连续的, 所以也要
        r_out, h_state = self.rnn(x, h_state)
