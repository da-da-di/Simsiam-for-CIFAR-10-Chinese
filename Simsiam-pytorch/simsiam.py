import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *


# 计算负余弦相似度（L2正则化后的MSE)
class NegativeCosineSimilarity(nn.Module):
    def __init__(self,
                 mode: str = 'simplified'
                 ) -> None:
        super(NegativeCosineSimilarity, self).__init__()

        self.mode = mode
        assert self.mode in ['simplified', 'original'], \
            'loss mode must be either (simplified) or (original)'

    # 用论文中伪代码的方法计算负余弦相似度
    def _forward1(self,
                  p: Tensor,
                  z: Tensor,
                  ) -> Tensor:
        # 截断反向传播
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        loss = -(p * z).sum(dim=1).mean()
        return loss

    # 直接调用余弦相似度函数进行计算
    def _forward2(self,
                  p: Tensor,
                  z: Tensor,
                  ) -> Tensor:
        # 截断反向传播
        z = z.detach()
        loss = - F.cosine_similarity(p, z, dim=-1).mean()
        return loss

    # 计算损失函数
    def forward(self,
                p1: Tensor,
                p2: Tensor,
                z1: Tensor,
                z2: Tensor,
                ) -> Tensor:
        # 自己实现的损失函数，和论文中的伪代码实现一致
        if self.mode == 'original':
            loss1 = self._forward1(p1, z2)
            loss2 = self._forward1(p2, z1)
            loss = loss1 / 2 + loss2 / 2
            return loss
        # 调用库函数直接计算
        elif self.mode == 'simplified':
            loss1 = self._forward2(p1, z2)
            loss2 = self._forward2(p2, z1)
            loss = loss1 / 2 + loss2 / 2
            return loss


# 3层的projectionMLP
class ProjectionMLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 2048,
                 output_dim: int = 2048,
                 ) -> None:
        super(ProjectionMLP, self).__init__()
        # 以下三层都包括一个全连接层和一个BN，一个Relu，最后一个输出层不包括Relu
        self.layer1 = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True)
                                    )

        self.layer2 = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True)
                                    )

        self.layer3 = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim)
                                    )

    # 前向传播
    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# predictorMLP有两层
class PredictionMLP(nn.Module):
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 512,
                 output_dim: int = 2048,
                 ) -> None:
        super(PredictionMLP, self).__init__()
        # 全连接，BN，Relu
        self.layer1 = nn.Sequential(nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU(inplace=True)
                                    )
        # 输出层只有全连接层
        self.layer2 = nn.Sequential(nn.Linear(in_features=hidden_dim, out_features=output_dim))
    # 前向传播
    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)

        return x


# 把传入的model改成适用于本任务的encoder模型
class EncodProject(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 hidden_dim: int = 2048,
                 output_dim: int = 2048
                 ) -> None:
        super(EncodProject, self).__init__()
        # 提取除了最去一层的模型
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        # 建立MLP作为最后一层
        self.projector = ProjectionMLP(input_dim=nn.Sequential(*list(model.children()))[-1].in_features,
                                       hidden_dim=hidden_dim,
                                       output_dim=output_dim
                                       )

    def forward(self, x: Tensor) -> Tensor:
        # resNet网络
        x = self.encoder(x)
        # 将张量x沿着维度1展平。即沿着第一个维度的所有数据连接起来，形成一个一维张量
        x = torch.flatten(x, 1)
        # 放入MLP层
        x = self.projector(x)
        return x


# Simsiam模型
class SimSiam(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 projector_hidden_dim: int = 2048,
                 projector_output_dim: int = 2048,
                 predictor_hidden_dim: int = 512,
                 predictor_output_dim: int = 2048
                 ) -> None:
        super(SimSiam, self).__init__()
        # 初始化两个encoder的模型
        self.encode_project = EncodProject(model,
                                           hidden_dim=projector_hidden_dim,
                                           output_dim=projector_hidden_dim
                                           )
        # 初始化predictor的模型
        self.predictor = PredictionMLP(input_dim=projector_output_dim,
                                       hidden_dim=predictor_hidden_dim,
                                       output_dim=predictor_output_dim)

    # 前向传播
    def forward(self,
                x1: Tensor,
                x2: Tensor
                ) -> Tuple[Tensor]:
        f, h = self.encode_project, self.predictor
        # 首先得到两个视角的图像经过encoder的编码结果
        z1, z2 = f(x1), f(x2)
        # 再通过preditor得到这两个编码的预测
        p1, p2 = h(z1), h(z2)

        return {'p1': p1,
                'p2': p2,
                'z1': z1,
                'z2': z2}
