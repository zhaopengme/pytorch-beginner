import torch
from torch import nn, optim
import torch.nn.functional as F

from torch.autograd import Variable

from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets

# 定义超参数
batch_size = 32
learning_rate = 1e-3
nump_epoches = 100 # 迭代训练次数

# 下载 mnist 数据
# train 表示是否为训练/测试数据
# transform
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), target_transform=None,
                               download=True)

test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), target_transform=None,
                              download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


