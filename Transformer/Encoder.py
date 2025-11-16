import torch.nn as nn
from MultiheadAttention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden_dim, dropout=0.1):
        """
        合并之前的内容，构建Encoder块
        :param d_model:
        :param n_heads:
        :param ffn_hidden_dim:
        :param dropout:
        """
        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        计算注意力分数 -> 一次dropout -> 残差连接后层归一化 ->
        前馈神经网络 -> 二次dropout -> 残差连接后层归一化 -> 输出
        （Encoder 一般不带有 mask）
        :param x:
        :param mask:
        :return:
        """
        attn_output = self.self_attn(x, mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.ln1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.ln2(x)

        return x