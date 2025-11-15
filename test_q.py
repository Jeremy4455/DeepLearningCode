import torch

# 基础用法
# 3D张量转置
x_3d = torch.arange(24).view(2,3,4)  # [2, 3, 4]
print(f"原始3D形状: {x_3d.shape}\n{x_3d}")

# 交换不同维度
x_t1 = x_3d.transpose(0, 1)  # [3, 2, 4]
x_t2 = x_3d.transpose(1, 2)  # [2, 4, 3]
print(f"交换(0,1): {x_t1.shape}\n {x_t1}")
print(f"交换(1,2): {x_t2.shape}\n {x_t2}")