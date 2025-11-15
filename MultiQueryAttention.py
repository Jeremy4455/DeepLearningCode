import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        Multi-Head Attention的一种优化
        核心思想是所有查询头共享同一键值对，极大优化推理时的KV Cache显存占用
        :param d_model:
        :param n_heads:
        """
        assert d_model % n_heads == 0
        super(MultiQueryAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)

        self.W_K = nn.Linear(d_model, self.d_k)
        self.W_V = nn.Linear(d_model, self.d_k)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # 输入 x 的维度: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()

        # 线性变换得到 Q, K, V
        Q = self.W_Q(x)  # [batch_size, seq_len, d_model]
        K = self.W_K(x)
        V = self.W_V(x)

        # Q处理不变
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # KV被投影为单头
        K = K.view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, 1, self.d_k).transpose(1, 2)

        # 利用Pytorch广播机制来实现
        scores = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

        # 同样利用广播机制实现V的乘法
        output = torch.matmul(attention, V)

        # W_O 的作用：将多头输出合并并投影回原始维度，contiguous()保证输出在内存中连续
        # 展平最后两个维度: [batch_size, seq_len, n_heads * d_k] = [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(output)
        return output


if __name__ == "__main__":
    batch_size = 6
    n_heads = 8
    d_model = 512
    seq_len = 10

    x = torch.randn(batch_size, seq_len, d_model)

    mqa = MultiQueryAttention(d_model, n_heads)

    no_mask = mqa(x)
    print(no_mask.shape)

    casual_mask = torch.ones(seq_len, seq_len)
    casual_mask = torch.triu(casual_mask)
    with_mask = mqa(x, mask=casual_mask)
    print(with_mask.shape)