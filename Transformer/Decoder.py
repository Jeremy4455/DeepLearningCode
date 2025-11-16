import torch.nn as nn
from MultiheadAttention import MultiHeadAttention


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden_dim, dropout=0.1):
        """
        Decoder比Encoder多加一次交叉注意力机制
        :param d_model:
        :param n_heads:
        :param ffn_hidden_dim:
        :param dropout:
        """
        super(TransformerDecoder, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        Decoder 带有掩码的自注意力机制
        交叉注意力：让Decoder能够"查看"Encoder学到的输入序列信息
        :param x:
        :param mask:
        :return:
        """
        self_attn_output = self.self_attn(x, mask=self_attn_mask)
        x = self.ln1(x + self.dropout1(self_attn_output))

        # 这里实际上需要把encoder_output作为KV输入到MHA中
        # cross_attn_mask = self.cross_attn(q=x, k=encoder_output, v=encoder_output, mask=cross_attn_mask)
        cross_attn_output = self.cross_attn(x, mask=cross_attn_mask)
        x = self.ln2(x + self.dropout2(cross_attn_output))

        ffn_output = self.ffn(x)
        x = self.ln3(x + self.dropout3(ffn_output))
        return x
