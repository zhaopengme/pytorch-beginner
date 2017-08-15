import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np

import time

# batch_size = 32
# learning_rate = 0.01
# num_epoches = 50
# use_gpu = torch.cuda.is_available()
#
# train_dataset = datasets.MNIST(
#     root='./data',
#     train=True,
#     transform=transforms.ToTensor(),
#     download=True
# )
#
# test_dataset = datasets.MNIST(
#     root='./data',
#     train=False,
#     transform=transforms.ToTensor(),
#     download=True
# )


x = np.random.rand(3,3)
print(x)
print('*' *20)
print(x[-1])