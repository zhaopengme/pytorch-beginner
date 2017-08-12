import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import time

batch_size = 32
learning_rate = 0.01
num_epoches = 50
use_gpu = torch.cuda.is_available()

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


class Convolutional(nn.Module):
    def __init__(self, n_class):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1,28,28) 图片是黑白色的,通道就是1,图片大小是28x28
            nn.Conv2d(
                in_channels=1,  # 图片的厚度,由于是黑白的,所以厚度是1,如果是彩色的,那就是 RGB, 就是3
                out_channels=16,  # 输出图片的厚度,由平面转为立体 自己定义
                kernel_size=5,  # 过滤框/卷积器的大小,即5x5的大小,可以写成(3,4)
                stride=1,  # 过滤框/卷积器的移动步长
                padding=2  # 对图片周围进行填充, padding=(kernel_size-1)/2 当 stride=1
            ),  # out shape (16,28,28) 图片的厚度变成16,大小没有变化,还是28x28
            nn.ReLU(),  # 激活函数  不改变大小
            nn.MaxPool2d(kernel_size=2)  # 过滤框/卷积器的大小,即2x2的大小,可以写成(3,4),在2x2的平面区域中采样,取这个平面区域中的最大值,所以结果会变形,变为(16,14,14)
        )

        self.conv2 = nn.Sequential(  # input shape (16,14,14)
            nn.Conv2d(
                in_channels=16,  # 第一层卷积后图片厚度已经变成了16,此处接受就是16
                out_channels=32,  # 输出图片的厚度,由平面转为立体 自己定义
                kernel_size=5,  # 过滤框/卷积器的大小,即5x5的大小,可以写成(3,4)
                stride=1,  # 过滤框/卷积器的移动步长
                padding=2  # 对图片周围进行填充, padding=(kernel_size-1)/2 当 stride=1
            ),  # out shape (32,14,14)
            nn.ReLU(),  # 激活函数  不改变大小
            nn.MaxPool2d(kernel_size=2)  # out shape (32,7,7)
        )

        self.out = nn.Linear(32 * 7 * 7, n_class)  #

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将多维的卷积图,展开为(batch_size,32*7*7)
        out = self.out(x)
        return out


model = Convolutional(10)  # 10 个分类

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
        # imgSize = img.size(0)  # 矩阵中第0维的大小,如果不带参数,就是矩阵的大小
        # img = img.view(imgSize, -1)  # 将图片展开成 28x28  ,view 类似于 reshape的用法,改变数组/矩阵的形状 -1表示自动分配计算
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
