import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttentionWithCache(nn.Module):
    def __init__(self, d_model, n_heads):
        """
        手搓多头带 KV Cache 的自注意力机制
        :param d_model: 模型维度
        :param n_heads: 注意力头数量
        """
        super(MultiHeadAttentionWithCache, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, past_key_value=None):
        # 输入 x 的维度: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()

        # 线性变换得到 Q, K, V
        Q = self.W_Q(x)  # [batch_size, seq_len, d_model]
        K = self.W_K(x)  # [batch_size, seq_len, d_model]
        V = self.W_V(x)  # [batch_size, seq_len, d_model]

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 现在K，V的维度 [batch_size, num_heads, seq_len + past_seq_len, d_k]
            K = torch.cat((K, past_value), dim=2)
            V = torch.cat((V, past_value), dim=2)
        present_key_value = (K, V)

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
        return output, present_key_value


if __name__ == '__main__':
    d_model = 512
    num_heads = 8
    mha = MultiHeadAttentionWithCache(d_model, num_heads)
    # --- 阶段一：预填充（Pre-fill） ---
    prefill_len = 5
    x_prefill = torch.randn(1, prefill_len, d_model)
    print("--- 阶段一：预填充（Pre-fill） ---")
    prefill_mask = torch.tril(torch.ones(prefill_len, prefill_len)).unsqueeze(0).unsqueeze(1)
    output_prefill, kv_cache = mha(x_prefill, mask=prefill_mask, past_key_value=None)
    print(f"预填充输入形状: {x_prefill.shape}")
    print(f"预填充输出形状: {output_prefill.shape}")
    print(f"预填充后 K cache 形状: {kv_cache[0].shape}")
    print(f"预填充后 K cache 序列长度: {kv_cache[0].size(2)}")
    print("----------------------------------------\n")

    # --- 阶段二：解码（Decoding）---
    decode_len = 1
    x_decode = torch.randn(1, decode_len, d_model)
    print("--- 阶段二：解码（Decoding） ---")
    
    # 掩码只针对新 token 的注意力计算
    # `scores` 的形状是 (batch, num_heads, 1, total_seq_len)
    # 我们需要的掩码形状是 (1, 1, 1, total_seq_len)
    past_len = kv_cache[0].size(2)
    new_len = past_len + decode_len
    
    # 创建一个下三角矩阵作为完整的掩码，然后只取最后一行
    decode_mask = torch.tril(torch.ones(new_len, new_len))[-decode_len:, :].unsqueeze(0).unsqueeze(1)
    
    output_decode, new_kv_cache = mha(x_decode, mask=decode_mask, past_key_value=kv_cache)

    print(f"解码输入形状: {x_decode.shape}")
    print(f"解码输出形状: {output_decode.shape}")
    print(f"解码后 K cache 形状: {new_kv_cache[0].shape}")
    print(f"解码后 K cache 序列长度: {new_kv_cache[0].size(2)}")
    print("----------------------------------------\n")