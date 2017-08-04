import torch
from torch import nn, optim
from torch.autograd import Variable

import numpy as np
import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

iterations = 1000

for iteration in range(iterations):
    inputs = Variable(x_train)
    targets = Variable(y_train)

    out = model(inputs)
    loss = criterion(out, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.data[0])

model.eval()

out = model(Variable(x_train))

result = out.data.numpy()

print(result)
