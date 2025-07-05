import torch
import torch.nn as nn

class VGGTAdapter(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.head_proj = nn.Linear(512, d_model)
        self.front_proj = nn.Linear(512, d_model)
        self.vggt_proj = nn.Linear(2048, d_model)

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)

        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, head_feat, front_feat, vggt_feat):
        # head/front: [B, 512] -> [B, 512]
        head_feat_proj = self.head_proj(head_feat)
        front_feat_proj = self.front_proj(front_feat)

        # vggt_feat: [B, 8, 2048] -> [B, 8, 512]
        vggt_feat_proj = self.vggt_proj(vggt_feat)

        # stack head and front as query: [B, 2, 512]
        query = torch.stack([head_feat_proj, front_feat_proj], dim=1)

        # cross-attention
        attn_output, _ = self.cross_attn(query, vggt_feat_proj, vggt_feat_proj)  # [B, 2, 512]

        # flatten and fuse
        fusion = attn_output.reshape(attn_output.shape[0], -1)  # [B, 1024]
        fusion = self.fusion_fc(fusion)  # [B, 512]
        return fusion
