import torch
from  torch.autograd import Variable
from  torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, datasets

# 定义超参数
batch_size = 64  # 每次训练样本的数量
learning_rate = 0.001  # 学习率
num_epoches = 20  # 训练次数

# 下载 mnist 数据
train_dataset = datasets.MNIST(
    './mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    './mnist',
    train=False,
    transform=transforms.ToTensor()
)

train_DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,  # 图片每行的数据像素点,输入序列的数据数量
            hidden_size=64,  # rnn hidden unit ,输出维度
            num_layers=1,  # 有几层 rnn
            batch_first=True  # input & out 会以 batch size 为第一维度的特征集, e.g. (batch,time_step,input_size)
        )
