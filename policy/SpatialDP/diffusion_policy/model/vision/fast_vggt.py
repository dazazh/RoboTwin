import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGTPredictor(nn.Module):
    def __init__(self, image_feat_dim=512, vggt_feat_dim=2048, vggt_seq_len=8, hidden_dim=1024, num_layers=3):
        super().__init__()

        # 图像编码器（resnet18，frozen）
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # 去掉最后fc
        self.head_encoder = nn.Sequential(*modules)
        self.front_encoder = nn.Sequential(*modules)

        # 冻结参数
        for param in self.head_encoder.parameters():
            param.requires_grad = False
        for param in self.front_encoder.parameters():
            param.requires_grad = False

        # VGG-T feat 映射到 image_feat_dim
        self.vggt_proj = nn.Linear(vggt_feat_dim, image_feat_dim)

        # 融合 + MLP
        input_dim = image_feat_dim * 2 + image_feat_dim  # head + front + vggt_proj
        mlp_layers = []
        for i in range(num_layers):
            mlp_layers.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)

        # 最后映射到 vggt_feat_t1 (8, 2048)
        self.output_layer = nn.Linear(hidden_dim, vggt_seq_len * vggt_feat_dim)

    def forward(self, batch):
        # batch 是 dict，包含 batch_size 张量
        head_img = batch['head_img']     # (B, 3, H, W)
        front_img = batch['front_img']    # (B, 3, H, W)
        vggt_feat_t0 = batch['vggt_feat_t0']  # (B, 8, 2048)

        head_feat = self.head_encoder(head_img).squeeze(-1).squeeze(-1)  # (B, 512)
        front_feat = self.front_encoder(front_img).squeeze(-1).squeeze(-1)

        B, L, D = vggt_feat_t0.shape
        vggt_feat_proj = self.vggt_proj(vggt_feat_t0).mean(dim=1)  # (B, 512)

        x = torch.cat([head_feat, front_feat, vggt_feat_proj], dim=-1)  # (B, 512*3=1536)
        x = self.mlp(x)
        out = self.output_layer(x)  # (B, 8*2048)
        out = out.view(B, L, D)
        return out

