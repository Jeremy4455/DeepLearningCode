import torch


def freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    计算 RoPE 所需要的频率张量
    :param dim: 头的维度（偶数）
    :param end: 序列最大长度
    :param theta:
    :return:
    """
    # 计算基础频率：θ^(-2i/d)，i=0,1,2,...,dim/2-1
    freq_base = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 位置索引：0,1,2,...,end-1
    t = torch.arange(end, dtype=torch.float32)

    # 外积：每个位置 × 每个频率
    freqs = torch.outer(t, freq_base)

    # 创建复数：幅度为1，相位为freqs
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    seq_len = x.shape[1]
    rotary_factor = freqs_cis[:seq_len].view(1, seq_len, 1, -1)
    x_rotated = x_complex * rotary_factor

    x_out = torch.view_as_real(x_rotated).flatten(3)

    return x_out.type_as(x)


if __name__ == '__main__':
    d_model = 64
    max_len = 128
    n_heads = 4
    head_dim = d_model // n_heads

    # 1. 预计算频率
    freqs_cis = freqs_cis(dim=head_dim, end=max_len)

    # 2. 模拟 Q/K 向量: (B, S, H, D)
    xq = torch.randn(2, 50, n_heads, head_dim)

    # 3. 应用 RoPE
    xq_rotated = apply_rotary_emb(xq, freqs_cis)

    print(f"旋转 Q' 形状: {xq_rotated.shape}")