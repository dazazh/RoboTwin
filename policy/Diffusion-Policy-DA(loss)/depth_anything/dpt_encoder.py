import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from depth_anything.blocks import FeatureFusionBlock, _make_scratch

class DPT_DINOv2_Encoder(nn.Module):
    def __init__(self, encoder='vits', num_layers=4, localhub=True):
        super(DPT_DINOv2_Encoder, self).__init__()

        assert encoder in ['vits', 'vitb', 'vitl']

        # ✅ 加载 DINOv2 预训练模型（使用已有 checkpoint）
        if localhub:
            self.pretrained = torch.hub.load('./torchhub/facebookresearch_dinov2_main', 
                                             f'dinov2_{encoder}14', 
                                             source='local', 
                                             pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 
                                             f'dinov2_{encoder}14')

        # ✅ 设置 eval 模式（不会影响梯度）
        self.pretrained.eval()
        self.fc = nn.Linear(384, 256)

        # ✅ 选择提取的 Transformer 层数
        self.num_layers = num_layers

    def forward(self, x):
        with torch.no_grad():
            h, w = x.shape[-2:]

            # ✅ 仍然支持计算梯度
            features = self.pretrained.get_intermediate_layers(x, self.num_layers, return_class_token=False)
            # torch.mean(torch.stack(features, dim=0), dim=0)

        # 特征降维后处理
        # 1. 先拼接成一个 Tensor，形状变成 [batch_size, 4, 625, 384]
        features = torch.stack(features, dim=1)  # [batch_size, 4, 625, 384]

        # 2. 平均池化：对 625 个 patch 求平均，得到 [batch_size, 4, 384]
        features = features.mean(dim=2)

        # 3. 降维：用一个全连接层将 384 维映射到 256 维
        features = self.fc(features)  # [batch_size, 4, 256]

        # 4. 再求平均，变成最终的 [batch_size, 256]
        features = features.mean(dim=1)  # [batch_size, 256]

        # print(features.shape)  # torch.Size([batch_size, 256])

        return features  # 直接返回 Transformer 特征