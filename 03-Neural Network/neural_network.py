__author__ = 'SherlockLiao'

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time

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
    shuffle=True  # 每次训练取数据的时候,是否随机打乱顺序
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义神经网络模型
# 定义为简单三层结构
class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# in_dim 数据的维度
# n_class 分类数量


model = NeuralNetwork(28 * 28, 500, 200, 10)  # 图片大小是28x28
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降

# 开始训练
for epoch in range(num_epoches):
    startTime = time.time()

    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0  # 本次训练累加损失函数的值
    running_acc = 0.0  # 本次训练累加正确率的值

    # 每次都是根据 batch_size 来小批量处理的,要注意累加和乘以总数
    for i, data in enumerate(train_loader, 1):
        img, label = data  # train data 里面包含有两部分数据库,一部分是处理后的图片数据,一部分是表情 label 数据
        imgSize = img.size(0)  # 矩阵中第0维的大小,如果不带参数,就是矩阵的大小
        img = img.view(imgSize, -1)  # 将图片展开成 28x28  ,view 类似于 reshape的用法,改变数组/矩阵的形状 -1表示自动分配计算
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # 向前传播
        out = model(img)  # out 计算我们预测值 ?? 这里描述的不准确,可以认为就是预测出来了一组数据
        loss = criterion(out, label)  # 计算损失函数/ loss/误差 比较预测的值和实际值的误差
        temp_loss = loss.data[0] * label.size(0)  # loss 是个 variable，所以取 data，因为 loss 是算出来的平均值，所以乘上数量得到总的
        running_loss = running_loss + temp_loss  # temp_loss 是计算的本次小批量的值,要累加才是合计的

        # torch.max  返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引。
        # 这里 1 表示取列上面的最大值,如果是0 就会按照行来取值
        # pred 是索引的位置,一共有10个索引位置,算出来哪个位置的概率最大,这个值就是这个索引位置
        _, pred = torch.max(out, 1)

        num_correct = (pred == label).sum()  # (pred == label) 比较相同位置上面值,如果相等就是1,否则就是0. sum 之后,就算出来了有多少个正确的了
        running_acc = running_acc + num_correct.data[0]  # 累计正确的数量

        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:  # 每300次打印一下
            l = batch_size * i  # 样本数量
            tempRunningLoss = running_loss / l  # 当前损失函数的值 合计/样本数量
            tempRunningAcc = running_acc / l  # 当前的正确率 正确的数量/样本数量

            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, num_epoches, tempRunningLoss, tempRunningAcc))

    length = len(train_dataset)  # 所有的样本数量
    finishRunningLoss = running_loss / length  # 最终损失函数的值 合计/样本数量
    finishRunningAcc = running_acc / length  # 最终的正确率 正确的数量/样本数量

    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(epoch + 1, finishRunningLoss, finishRunningAcc))

    endTime = time.time()

    print('consuming time:{}'.format(endTime - startTime))

# 修改为测试验证模式
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
    eval_loss = eval_loss + loss.data[0] * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc = eval_acc + num_correct.data[0]

testLength = len(test_dataset)  # 测试样本的数量
testLoss = eval_loss / testLength
testAcc = eval_acc / testLength

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(testLoss, testAcc))
print()

# 保存模型
torch.save(model.state_dict(), './logstic.pth')
