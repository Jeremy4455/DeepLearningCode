import math
import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        绝对位置编码手搓
        :param d_model: 模型维度
        :param max_len: 最大序列长度
        """
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        # torch.arange(0, max_len): 生成 [0, 1, 2, ..., 4999]
        #
        # .unsqueeze(1): 从形状 (5000,) 变为 (5000, 1)
        #
        # 结果：[[0], [1], [2], ..., [4999]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 步骤分解：
        # torch.arange(0, d_model, 2)  # [0, 2, 4, ..., d_model-2] 偶数索引
        # (-math.log(10000.0) / d_model)  # -log(10000)/d_model
        # 两者相乘后取exp: torch.exp(...)  # 计算频率项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        为输入张量添加位置编码
        :param x:输入张量(batch_size, seq_len, d_model)
        :return:
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


if __name__ == '__main__':
    pos_encoder = PositionEncoding(d_model=4, max_len=10)

    # 查看位置编码
    print("位置编码矩阵形状:", pos_encoder.pe.shape)  # (1, 10, 4)
    print("\n前3个位置的位置编码:")
    print(pos_encoder.pe[0, :3, :])

    # 测试前向传播
    x = torch.randn(2, 5, 4)  # (batch_size=2, seq_len=5, d_model=4)
    print(f"\n输入形状: {x.shape}")
    print(x)
    output = pos_encoder(x)
    print(f"输出形状: {output.shape}")
    print(output)