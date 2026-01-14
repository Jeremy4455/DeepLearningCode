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
    """
    safe-softmax -> online-softmax -> Flash Attention
    核心思想，把QKV矩阵进行tiling，即分块，使得每个分块能够在GPU的SRAM上进行计算
    减少对HBM的读写次数，从而加速Attention计算速度。

    safe-softmax: 防止x过大，对每一项减去max(x)
    online-softmax: 边看边更新max(x)和softmax_sum
    """
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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            attn_mask: 目前未使用（可扩展支持 causal / padding mask）

        Returns:
            输出张量 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 一次性投影得到 Q/K/V
        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑成多头形式：(B, nh, N, dh)
        def split_heads(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(q)  # (B, nh, N, dh)
        K = split_heads(k)
        V = split_heads(v)

        # 初始化输出累加器
        # out 将直接累加  (softmax后的概率 × V_block)
        out = torch.zeros_like(Q)  # (B, nh, N, dh)

        # 统计量：用于实现在线 softmax
        # max_scores: 每行目前见过的最大 logit 值   (B, nh, N, 1)
        # softmax_sum: 每行目前累积的  exp(logit - max) 之和   (B, nh, N, 1)
        max_scores = torch.full((batch_size, self.num_heads, seq_len, 1),
                                float("-inf"), device=x.device)
        softmax_sum = torch.zeros((batch_size, self.num_heads, seq_len, 1),
                                  device=x.device)

        block_size = 64  # 分块大小（可调，典型 32~128，视显卡 SRAM 大小）

        # 外层循环：按 K/V 的列（序列维度）分块
        for i in range(0, seq_len, block_size):
            i_end = min(i + block_size, seq_len)
            K_block = K[:, :, i:i_end]  # (B, nh, block_size, dh)
            V_block = V[:, :, i:i_end]  # (B, nh, block_size, dh)

            # 内层循环：按 Q 的行（序列维度）分块
            for j in range(0, seq_len, block_size):
                j_end = min(j + block_size, seq_len)
                Q_block = Q[:, :, j:j_end]  # (B, nh, block_size, dh)

                # 计算当前块的注意力 logits
                # attn_chunk: (B, nh, block_size, block_k_size)
                attn_chunk = torch.matmul(Q_block, K_block.transpose(-2, -1)) * self.scaling

                # ------------------ 在线 softmax 核心部分 ------------------

                # 1. 当前块的最大值（每行）
                M_ij = attn_chunk.max(dim=-1, keepdim=True)[0]  # (B, nh, block_size, 1)

                # 2. 更新全局最大值（取 max）
                L_j_old = max_scores[:, :, j:j_end]  # 旧的最大值
                L_j_new = torch.maximum(L_j_old, M_ij)  # 新的最大值

                # 3. exp 修正因子：旧的统计量要缩放到新的 max 基准下
                exp_correction_factor = torch.exp(L_j_old - L_j_new)  # <=1

                # 4. 当前块的 exp 值（基于新的全局 max 缩放）
                exp_scores = torch.exp(attn_chunk - L_j_new)  # (B,nh,block_q,block_k)

                # 5. 当前块的 softmax 分母增量
                softmax_sum_chunk = exp_scores.sum(dim=-1, keepdim=True)

                # 6. 更新全局分母
                #    新分母 = 旧分母 × 修正因子 + 当前块的贡献
                softmax_sum[:, :, j:j_end] = (
                        softmax_sum[:, :, j:j_end] * exp_correction_factor
                        + softmax_sum_chunk
                )

                # 7. 更新全局 max
                max_scores[:, :, j:j_end] = L_j_new

                # ------------------ 累加到输出上 ------------------

                # 8. 先把旧的输出缩放到新的基准（乘以修正因子）
                out[:, :, j:j_end] *= exp_correction_factor

                # 9. 加上当前块的贡献：  (exp_scores / 当前块分母) × V_block
                #    但因为我们还没有除以总分母，所以这里直接累加分子部分
                out[:, :, j:j_end] += torch.matmul(exp_scores, V_block)

                # 注意：这里没有显式除以 softmax_sum，因为最后统一除

        # 最后一步：归一化（全局分母）
        # out 现在是  Σ (exp(qk - global_max) * v)  的累加
        # 除以  Σ exp(qk - global_max)  就得到正确结果
        out = out / softmax_sum.clamp(min=1e-6)  # 防止除零

        # 恢复形状并输出
        out = out.transpose(1, 2).contiguous()  # (B, N, nh, dh)
        out = out.view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(out)