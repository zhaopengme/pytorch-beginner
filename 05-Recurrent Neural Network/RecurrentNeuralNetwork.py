import torch
from  torch.autograd import Variable
from  torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, datasets

# 定义超参数
batch_size = 64  # 每次训练样本的数量
learning_rate = 0.001  # 学习率
num_epoches = 2  # 训练次数

is_gpu = torch.cuda.is_available()

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
        )  # 输出 (64)

        self.linear = nn.Linear(
            64,  # rrn 输出是64
            10  # 10 个分类作为输出
        )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, temp_step, out_size)
        # h_n (n_layers, batch, hidden_size) LSTM 有两个 hidden state, h_n 是主线, h_c 是支线
        # h_c (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        temp_data = r_out[:, -1, :]
        out = self.linear(temp_data)
        return out


model = Rnn()
if (is_gpu):
    model = model.cuda()

# 定义损失函数和优化函数
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练

for epoch in range(num_epoches):
    print('*' * 30)
    print('epoch ' + str(epoch))
    running_losss = 0.0  # 损失
    running_acc = 0.0  # 正确率
    for step, data in enumerate(train_DataLoader, 1):  # 下标冲1开始,方便下面求余
        img, label = data  # data 里面包含 图片和 label 两部分数据,每种都有 batch_size 个,即去的样本数量
        # img shpae (64,1,28,28) 需要转换成需要的(batch, time_step, input_size) 形式
        tempImg = img.view(-1, 28, 28)  # tempImage shape (64,28,28)
        x = Variable(tempImg)
        y = Variable(label)
        if (is_gpu):
            x = x.cuda()
            y = y.cuda()

        # 前向传播
        out = model(x)  # out.data shape (64,10)
        loss = loss_func(out, y)  # loss.data 是一个在本次根据样本数据训练后,预测的结果和真实值之间的数值,注意!!!!,这是个平均值,要算本次预测的误差,要乘以样本数量
        temp_loss = loss.data[0] * label.size(0)  # 本次小批量训练的损失值,胡差值
        running_losss = running_losss + temp_loss  # 累加得到多少油批次训练的损失值
        _, pre_data = torch.max(out, 1)  # 获取预测出来的值
        acc_num = (pre_data == y).sum()  # 将预测出来的值和真实值进行对比,并 sum 计算预测的数量
        running_acc = running_acc + acc_num.data[0]  # 累加预测准确的数量

        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:  # 每100次打印一下

            l = step * batch_size  # 样本数量
            temp_loss_p = running_losss / l  # 损失率
            temp_acc_p = running_acc / l  # 准确率

            print("step:{} loss:{} acc:{}".format(step, temp_loss_p, temp_acc_p))

    # 本次训练换成之后,在打印一次训练
    l = len(train_dataset)  # 样本数量
    temp_loss_p = running_losss / l  # 损失率
    temp_acc_p = running_acc / l  # 准确率
    print('*' * 30)
    print("finalloss:{} acc:{}".format(temp_loss_p, temp_acc_p))

# 修改为测试验证模式
model.eval()

# 训练完成后,开始测试
model.eval()
print('-' * 30)

test_loss = 0.
test_acc = 0.
for step, data in enumerate(test_DataLoader, 1):
    img, label = data
    tempImg = img.view(-1, 28, 28)
    x = Variable(tempImg)
    y = Variable(label)
    if (is_gpu):
        x = x.cuda()
        y = y.cuda()

    out = model(x)
    loss = loss_func(out, y)
    temp_loss = loss.data[0] * label.size(0)
    _, pre_data = torch.max(out, 1)

    test_loss = test_loss + temp_loss
    temp_acc = (pre_data == y).sum()
    test_acc = test_acc + temp_acc.data[0]

    if step % 100 == 0:
        l = step * batch_size
        cur_loss = test_loss / l
        cur_acc = test_acc / l
        print("step:{} cur_loss:{} cur_acc:{}".format(step, cur_loss, cur_acc))

l = len(test_dataset)
test_loss_p = test_loss / l
test_acc_p = test_acc / l
print('*' * 30)
print("final test_loss_p:{} test_acc_p:{}".format(test_loss_p, test_acc_p))
