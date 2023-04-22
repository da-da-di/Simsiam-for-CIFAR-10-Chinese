# -*- coding: UTF-8 -*-
"""
@Project :SimSiam-pytorch 
@File    :si.py
@Author  :周宇康
@Date    :2023/4/19 8:43 
@Desc  :
@Contact : 1815963968@qq.com
"""
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from simsiam import SimSiam, NegativeCosineSimilarity

epoch = 5
start_form_train = False


def augmentation(image):
    n = random.random()
    if n < 0.2:
        # 直接缩放全体像素
        return image * 0.9
    elif n < 0.4:
        # 图片旋转5*n度(即60-180度)
        return T.RandomRotation(degrees=300 * n + 60)(image)
    elif n < 0.6:
        # 亮度、对比度和饱和度调节
        return T.ColorJitter(brightness=(2, 2), contrast=(0.5, 0.5), saturation=(0.5, 0.5))(image)
    elif n < 0.8:
        # 使用高斯核对图像进行模糊变换,sigma在3-7之间
        return T.GaussianBlur(kernel_size=(3, 3), sigma=3 + n * 5)(image)
    # elif n < 0.8:
    #     # 正方形补丁随机应用在图像中
    #     add_random_boxes(image, n_k=5)
    else:
        # 向图像中加入高斯噪声
        image = add_noise(image, 0.1 + n * 0.3)
        return image


def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


# def add_random_boxes(img, n_k, size=2):
#     print(img.size())
#     for k in range(n_k):
#         y, x = np.random.randint(0, 32 - size, (2,))
#         img[:, :, y:y + size, x:x + size] = 0
#     return img


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

print("训练集的长度:{}".format(len(train_data)))
print("测试集的长度:{}".format(len(test_data)))

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=256)
test_dataloader = DataLoader(test_data, batch_size=256)

# 添加tensorboard可视化数据
writer = SummaryWriter('logs_tensorboard')

if start_form_train:

    # 使用预训练的resNet18作为两个encoderd的模型，他们共享模型结构和所有参数
    model = resnet18(pretrained=True).cuda()
    # 将resNet18传入Simsiam模型中用作encoder
    learner = SimSiam(model, projector_hidden_dim=1024,
                      projector_output_dim=1024,
                      predictor_hidden_dim=256,
                      predictor_output_dim=1024).cuda()
    # 构造Adam优化器用于参数更新
    opt = torch.optim.Adam(learner.parameters(), lr=0.001)
    # 负余弦相似度，我们希望这个值越接近-1越好
    criterion = NegativeCosineSimilarity().cuda()


    def sample_unlabelled_images():
        return torch.randn(20, 3, 256, 256)


    i = 1
    for _ in range(epoch):
        for data in train_dataloader:
            # 数据分开 一个是图片数据，一个是真实值，训练阶段targets直接丢弃
            images, targets = data
            images = images.cuda()  # 放到GPU上一会训练用
            images2 = augmentation(images)
            images2 = images2.cuda()  # 放到GPU上一会训练用
            p1, p2, z1, z2 = learner(images, images2).values()
            # 损失函数
            loss = criterion(p1, p2, z1, z2)
            # 先将梯度置为零
            opt.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            opt.step()
            if i % 20 == 0 or i == 1 or i == 5:
                print(f'第{i}个batch的loss为:{loss}')
            writer.add_scalar('训练集损失', loss.item(), i)
            i = i + 1
        print(f'第{_ + 1}个epoch结束，loss为:{loss}')
    # 保存模型
    torch.save(learner, 'model_pytorch/model_{}.pth'.format(epoch))

learner = torch.load('model_pytorch/model_10.pth').cuda()

learner.encode_project.projector.layer3 = nn.Sequential(
    nn.Linear(in_features=1024, out_features=128, bias=False),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=128, out_features=10, bias=False),
)

model = learner.encode_project.cuda()

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 冻结前面的特征层
for para in list(model.parameters())[:-1]:
    para.requires_grad = False

# 换用交叉熵损失函数
loss = nn.CrossEntropyLoss().cuda()

# 优化器
opt = torch.optim.SGD(model.parameters(), lr=0.005)

i = 1  # 用于绘制测试集的tensorboard
for _ in range(epoch):
    for data in train_dataloader:
        # 数据分开 一个是图片数据，一个是真实值
        images, targets = data
        images = images.cuda()  # 放到GPU上一会训练用
        targets = targets.cuda()
        # 拿到预测值
        output = model(images)
        # 计算损失值
        loss_in = loss(output, targets)
        # 先梯度清零
        opt.zero_grad()
        # 反向传播+更新
        loss_in.backward()
        opt.step()
        if i % 20 == 0 or i == 1 or i == 5:
            print(f'第{i}个batch的loss为:{loss_in}')
        writer.add_scalar('有监督训练集损失', loss_in.item(), i)
        i = i + 1

    sum_loss = 0
    accurate = 0
    with torch.no_grad():
        for data in test_dataloader:
            # 这里的每一次循环 都是一个minibatch  一次for循环里面有64个数据。
            images, targets = data
            images = images.cuda()
            targets = targets.cuda()
            output = model(images)
            loss_in = loss(output, targets)

            sum_loss += loss_in
            accurate += (output.argmax(1) == targets).sum()

        print('第{}轮测试集的正确率:{:.2f}%'.format(epoch, accurate / len(test_data) * 100))

        writer.add_scalar('测试集正确', accurate / len(test_data) * 100, i)

# 关闭tensorboard
writer.close()
