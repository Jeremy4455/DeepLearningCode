import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, elementwise_affine=True):
        """
        手搓RMSNorm
        :param d_model: 数据维度
        :param eps:
        :param elementwise_affine: 是否可学习
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # dim=-1 的含义：
        # 表示对最后一个维度计算均值
        # 对于不同形状的输入：
        #
        # 2D张量 (batch, features)：对特征维度求均值
        #
        # 3D张量 (batch, seq_len, features)：对特征维度求均值
        #
        # keepdim=True 的作用：
        # 保持输出张量的维度数不变，只在求均值的维度上变为1
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if self.elementwise_affine:
            return x_normed * self.weight

        return x_normed


if __name__ == '__main__':
    rms_norm = RMSNorm(d_model=4)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                       [2.0, 4.0, 6.0, 8.0]]])  # shape: (1, 2, 4)

    print("输入数据:")
    print(x)
    print("形状:", x.shape)

    output = rms_norm(x)
    print("\nRMSNorm 输出:")
    print(output)
