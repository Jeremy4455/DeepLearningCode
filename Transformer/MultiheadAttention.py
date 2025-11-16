import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        手搓多头自注意力机制
        :param d_model: 模型维度
        :param n_heads: 注意力头数量
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # 输入 x 的维度: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()

        # 线性变换得到 Q, K, V
        Q = self.W_Q(x)  # [batch_size, seq_len, d_model]
        K = self.W_K(x)  # [batch_size, seq_len, d_model]
        V = self.W_V(x)  # [batch_size, seq_len, d_model]

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 2:
                # mask 初始: [seq_len, seq_len] 或 [batch_size, seq_len]
                # 扩展为: [batch_size, n_heads, seq_len, seq_len]
                mask = mask.unsqueeze(0).unsqueeze(1)
            # 应用掩码，将掩码为0的位置设为负无穷
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

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

    mha = MultiHeadAttention(d_model, n_heads)

    no_mask = mha(x)
    print(no_mask.shape)

    casual_mask = torch.ones(seq_len, seq_len)
    casual_mask = torch.triu(casual_mask)
    with_mask = mha(x, mask=casual_mask)
    print(with_mask.shape)