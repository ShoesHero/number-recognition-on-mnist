

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision  # 包括了一些数据库，图片的数据库也包含了
import matplotlib.pyplot as plt
import time

# 定义超参数
EPOCH = 3
BATCH_SIZE = 100
LR = 0.03
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root=r'D:\python\minist',  # 存储路径
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 把下载的数据改成Tensor形式
    # 把(0-255)转换成(0-1)
    download=DOWNLOAD_MNIST  # 如果没下载就确认下载
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

# 准备测试集
test_data = torchvision.datasets.MNIST(
    root=r'D:\python\minist',  # 存储路径
    train=False,  # 提取出来的不是training data，是test data
    transform=torchvision.transforms.ToTensor(),  # 把下载的数据改成Tensor形式
    # 把(0-255)转换成(0-1)
    download=False  # 如果已经下载了，就用False)
)

# !!!!!!!!!!!!!!!!!!!!!!!1
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[
         :5000].cuda() / 255
# 把test_data换到0-1之间
test_y = test_data.test_labels[:5000].cuda()


# 建立CNN神经网路
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 图像的高度
                out_channels=16,  # filter的高度，提取出来16个特征放到后面去
                kernel_size=5,  # filter为5*5,
                stride=1,  # 扫描两个相邻区域之间的步长
                padding=2  # 在图片周围围上一圈0，使filter扫描的时候边缘不会出现不够的情况
                # padding = (kenrel_size-stride )/2 = (5-1)/2 = 2
            ),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层,筛选重要信息
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 卷积层
            # 前面输出16层，现在输入就是16层，输出就是32层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 池化层,筛选重要信息
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32, 64, 5, 1, 2),  # 卷积层
        #     # 前面输出16层，现在输入就是16层，输出就是32层
        #     nn.ReLU(),  # 激活函数
        #     nn.MaxPool2d(kernel_size=2),  # 池化层,筛选重要信息
        # )
        self.out = nn.Linear(32 * 7 * 7, 10)
        # 输出是0-9十个类别的分类
        # 图片维度(1,28,28) -->conv2d --> (16,28,28) --> padding --> (16,14,14)
        # -->(16,14,14) -->conv2d --> (32,14,14) --> padding --> (32,7,7)

    # 三维数据展平成2维数据
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)  # 打印结构
# !!!!!!!!!!!!!!!!!!!!
cnn.cuda()

# 优化器和loss
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 优化器
loss_func = nn.CrossEntropyLoss()  # 计算损失函数

# 训练过程
t_start = time.time()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
        # !!!!!!!!!!!!!!
        b_x = Variable(b_x).cuda()  # batch x
        b_y = Variable(b_y).cuda()  # batch y

        output = cnn(b_x)  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # 打印出来训练效果
        if step % 50 == 0:
            test_output = cnn(test_x)
            # !!!!!!!!!!!!!!!!!!
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = torch.sum(pred_y == test_y).float() / float(test_y.size(0))
            # 算括号里的是否等于，等于表示预测对了记一次，总共对的次数除以总数就是accuracy
            # .float()表示把int的tensor强制转换成float
            # 除号后面也是一个int型的数，float强制类型转换
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)

t_end = time.time()

# 拿测试集前十个数据测试一下效果
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
# !!!!!!!!!!!!!!!!!
# 在打印之前要变回到cpu上
test_y = test_y.cpu()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
print('time:', t_end - t_start)
