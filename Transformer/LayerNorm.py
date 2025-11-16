import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        """
        手搓LayerNorm

        :param normalized_shape:维度，对于不同类型样本不同，如图像可以是（channel, height, weight），文本可以是(embedding_dim,)
        :param eps:
        :param elementwise_affine:是否使用可学习参数
        """
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = normalized_shape

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape

        # 计算需要norm的维度，shape = (C,H,W), dims = (-3, -2, -1)
        dims = tuple(range(-len(self.normalized_shape), 0))

        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        x_normed = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            return x_normed * self.gamma + self.beta

        return x_normed


if __name__ == '__main__':
    batch_size, seq_len, embed_dim = 2, 5, 4
    text_data = torch.randn(batch_size, seq_len, embed_dim)
    print("文本数据形状:", text_data.shape)  # (2, 5, 4)

    # 文本LayerNorm - 只对词向量维度
    text_norm = LayerNorm((embed_dim,))
    output_text = text_norm(text_data)
    print("文本归一化后形状:", output_text.shape)  # (2, 5, 4)

    # 图像数据示例
    batch_size, channels, height, width = 2, 3, 4, 4
    image_data = torch.randn(batch_size, channels, height, width)
    print("图像数据形状:", image_data.shape)  # (2, 3, 4, 4)

    # 图像LayerNorm - 对通道和空间维度
    image_norm = LayerNorm((channels, height, width))
    output_image = image_norm(image_data)
    print("图像归一化后形状:", output_image.shape)  # (2, 3, 4, 4)
