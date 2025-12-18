import torch
import torch.nn as nn
import math


class LoRALayer(nn.modules):
    """
    LoRA 旁路模块
    """

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_normal_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        x = self.lora_A(x)
        x = self.lora_B(x)
        return x * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            alpha
        )

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(model, rank, alpha):
    """
    将所有线性层替换为LoRA
    :param model:
    :param rank:
    :param alpha:
    :return:
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            apply_lora_to_model(module, rank, alpha)
