import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接调整维度（当输入输出尺寸/通道不同时）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        return torch.relu(out)

class SpatialReducer(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 0: 初始下采样（16 → 64, 518 → 259）
        self.stage0 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 259x259
        )
        
        # Stage 1-4: 仿 ResNet 的 4 个阶段
        self.stage1 = self._make_stage(64, 64, stride=1, num_blocks=2)    # 259x259
        self.stage2 = self._make_stage(64, 128, stride=2, num_blocks=2)   # 130x130
        self.stage3 = self._make_stage(128, 256, stride=2, num_blocks=2)  # 65x65
        self.stage4 = self._make_stage(256, 512, stride=2, num_blocks=2)  # 7x7 (65/2^3 ≈ 7)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_stage(self, in_channels, out_channels, stride, num_blocks):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage0(x)      # [b, 16, 518, 518] → [b, 64, 259, 259]
        x = self.stage1(x)      # [b, 64, 259, 259] → [b, 64, 259, 259]
        x = self.stage2(x)      # [b, 64, 259, 259] → [b, 128, 130, 130]
        x = self.stage3(x)      # [b, 128, 130, 130] → [b, 256, 65, 65]
        x = self.stage4(x)      # [b, 256, 65, 65] → [b, 512, 7, 7]
        x = self.avgpool(x)     # [b, 512, 7, 7] → [b, 512, 1, 1]
        return x.flatten(1)     # [b, 512]