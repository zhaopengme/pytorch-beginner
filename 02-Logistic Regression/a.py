__author__ = 'SherlockLiao'

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# 定义超参数
batch_size = 32  # 每次拿出来多少个样本进行训练
learning_rate = 1e-3  # +1e-3表示1*10^-3，即0.001    e的负3次方是exp（3）

num_epoches = 100  # x训练次数

# mnist 的数据结构 http://blog.csdn.net/qq_32166627/article/details/62218072  在这个里面有说明,
# pytorch 已经对这些数据做了处理

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data',  # 数据保存的路径
    train=True,  # 是否为训练数据还是测试数据
    transform=transforms.ToTensor(),  # 把数据转化为 tensor 格式,便于数据使用存储
    download=True  # 是否下载,如果已经下载完成,会自动跳过
)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

# DataLoader 批量数据训练的包装类
train_loader = DataLoader(
    train_dataset,  # 处理使用的数据集
    batch_size=batch_size,  # 每次batch 加载的样本数量
    num_workers=2,  # 加载数据的使用几个线程
    shuffle=True # 每次训练取数据的时候,是否随机打乱顺序
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Logistic Regression 模型
# 定义逻辑回归模型
class Logstic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logstic_Regression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        return out

# in_dim 数据的维度
# n_class 分类数量

model = Logstic_Regression(28 * 28, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss() # 交叉熵
optimizer = optim.SGD(model.parameters(), lr=learning_rate) # 随机梯度下降

# 开始训练
for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data # train data 里面包含有两部分数据库,一部分是处理后的图片数据,一部分是表情 label 数据
        imgSize = img.size(0) # 矩阵中第0维的大小,如果不带参数,就是矩阵的大小
        img = img.view(imgSize, -1)  # 将图片展开成 28x28
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

    model.eval()

    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print()

# 保存模型
torch.save(model.state_dict(), './logstic.pth')
