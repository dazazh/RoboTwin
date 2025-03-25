# # 
# import zarr
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取 Zarr 存储
# zarr_path = "/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/policy/Diffusion-Policy-DA(loss)/data/tube_grasp_D435_10.zarr"
# depth_key = "data/head_depth"  # 访问整个 depth 数组
# output_path = "depth_map_zarr.png"  # PNG 文件保存路径

# # 打开 Zarr 存储
# root = zarr.open(zarr_path, mode='r')

# # 读取整个深度数组
# depth_data = np.array(root[depth_key])

# # 打印形状，检查数据维度
# print(f"深度数据形状: {depth_data.shape}")

# # 选择第一帧（如果数据是 [N, H, W]）
# if len(depth_data.shape) == 3:  # 形状可能是 (N, H, W)
#     depth_map = depth_data[0]  # 选择第 1 帧
# elif len(depth_data.shape) == 2:  # 形状 (H, W)，说明只有一张
#     depth_map = depth_data
# else:
#     raise ValueError(f"未知的深度数据形状: {depth_data.shape}")

# # 可视化并保存深度图
# plt.figure(figsize=(8, 6))
# plt.imshow(depth_map, cmap='viridis')  # 选择合适的颜色映射
# plt.colorbar(label="Depth Value")  # 显示颜色条
# plt.axis("off")

# # 保存为 PNG 文件
# plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
# plt.close()

# print(f"第一帧深度图已保存至 {output_path}")

import zarr
import numpy as np
import matplotlib.pyplot as plt
from depth_anything.dpt import DepthAnything
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

# 读取 Zarr 存储
zarr_path = "/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/policy/Diffusion-Policy-DA(loss)/data/tube_grasp_D435_10.zarr"
depth_key = "data/head_depth"  # 访问整个 depth 数组
output_path = "./test_depth/depth_map_gray.png"  # PNG 文件保存路径

# 打开 Zarr 存储
root = zarr.open(zarr_path, mode='r')

# 读取整个深度数组
depth_data = np.array(root[depth_key])

# 打印形状，检查数据维度
print(f"深度数据形状: {depth_data.shape}")

# 选择第一帧（如果数据是 [N, H, W]）
if len(depth_data.shape) == 3:  # 形状可能是 (N, H, W)
    depth_map = depth_data[0]  # 选择第 1 帧
elif len(depth_data.shape) == 2:  # 形状 (H, W)，说明只有一张
    depth_map = depth_data
else:
    raise ValueError(f"未知的深度数据形状: {depth_data.shape}")

# 计算深度值范围，增强对比度
vmin, vmax = np.percentile(depth_map, [5, 95])  # 去掉极端值以提升可视化效果

# 可视化并保存灰度深度图
plt.figure(figsize=(8, 6))
plt.imshow(depth_map, cmap='gray', vmin=vmin, vmax=vmax)  # 灰度图
plt.colorbar(label="Depth Value")  # 显示颜色条
plt.axis("off")

# 保存为 PNG 文件
plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close()

print(f"第一帧深度图（灰度）已保存至 {output_path}")

rgb_key = 'data/head_camera'
rgb_data = np.array(root[rgb_key])
image = np.transpose(rgb_data[0], (1, 2, 0))  # 变换为 (H, W, C)

plt.figure(figsize=(8, 6))
plt.imshow(image) 
plt.colorbar(label="rgb Value")  # 显示颜色条
plt.axis("off")

# 保存为 PNG 文件
plt.savefig("./test_depth/rgb.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close()

model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14").to("cuda:0")

this_resizer = nn.Identity()
h=350
w=350
this_resizer = torchvision.transforms.Resize(
    size=(h,w)
)
# input_shape = (shape[0],h,w)

# configure randomizer
this_randomizer = nn.Identity()
# configure normalizer
this_normalizer = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
image = torch.from_numpy(image).permute(2, 0, 1).float()
image = this_transform(image)

# 添加 batch 维度并移动到 `DEVICE`
image = image.unsqueeze(0).to("cuda:0")
with torch.no_grad():
    depth, _ = model(image)


depth = depth[0]
depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

depth = depth.cpu().numpy().astype(np.uint8)
plt.figure(figsize=(8, 6))
plt.imshow(depth,cmap='gray') 
plt.colorbar(label="rgb Value")  # 显示颜色条
plt.axis("off")

# 保存为 PNG 文件
plt.savefig("./test_depth/try_depth.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close()