import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# Pytorch2.0+内置FlashAttention
# out = F.scaled_dot_product_attention(
#     query=q,
#     key=k,
#     value=v,
#     attn_mask=attn_mask,
#     dropout_p=0.0,
#     is_causal=False,
# )

class FlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask:torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(q)
        K = split_heads(k)
        V = split_heads(v)

        out = torch.zeros_like(Q)
        block_size = 64

        device = x.device
        max_scores = torch.full((batch_size, self.num_heads, seq_len, 1), float("-inf"), device=device)
        softmax_sum = torch.zeros(batch_size, self.num_heads, seq_len, 1, device=device)

        for i in range(0, seq_len, block_size):
            K_block = K[:, :, i:i + block_size]
            V_block = V[:, :, i:i + block_size]

            for j in range(0, seq_len, block_size):
                Q_block = Q[:, :, j:j + block_size]

                q_slice = slice(j, j + block_size)

                attn_chunk = torch.matmul(Q_block, K_block.transpose(-2, -1)) * self.scaling

                M_ij = attn_chunk.max(dim=-1, keepdim=True)[0]

                L_j_old = max_scores[:, :, q_slice]
                L_j_new = torch.maximum(L_j_old, M_ij)

                exp_correction_factor = torch.exp(L_j_old - L_j_new)
                exp_scores = torch.exp(attn_chunk - L_j_new)

                softmax_sum_chunk = exp_scores.sum(dim=-1, keepdim=True)
                softmax_sum[:, :, q_slice] = (softmax_sum_chunk * exp_correction_factor) + softmax_sum_chunk

                max_scores[:, :, q_slice] = L_j_new

                out[:, :, q_slice] *= exp_correction_factor
                out[:, :, q_slice] += torch.matmul(exp_scores, V_block)

        out = out / softmax_sum
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)