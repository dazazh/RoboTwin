import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
import numpy as np
import matplotlib.pyplot as plt

from depth_anything.blocks import FeatureFusionBlock, _make_scratch
from peft import LoraConfig, get_peft_model

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out
        
        
class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vits', features=256, out_channels=[48, 96, 192, 384], use_bn=False, use_clstoken=False, localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl']

        # 加载 DINOv2 预训练模型
        if localhub:
            self.pretrained = torch.hub.load('./torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        # LoRA 适配 `qkv` 和 `proj` 层
        # lora_config = LoraConfig(
        #     r=16,  # 低秩维度
        #     lora_alpha=32,  # 缩放系数
        #     lora_dropout=0.1,  # Dropout
        #     target_modules=["qkv", "proj"],  # 只适配 Transformer 的注意力层
        #     bias="none"
        # )
        # self.pretrained = get_peft_model(self.pretrained, lora_config)

        # 获取 Transformer 维度
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        # 冻结 `depth_head` 以减少显存消耗
        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
        for param in self.depth_head.parameters():
            param.requires_grad = False  # 冻结 depth_head

        # 冻结 patch embedding 层
        for param in self.pretrained.patch_embed.parameters():
            param.requires_grad = False

        # 冻结前 8 层 Transformer block
        for i in range(11):
            for param in self.pretrained.blocks[i].parameters():
                param.requires_grad = False

        # for param in self.pretrained.parameters():
        #     param.requires_grad = False
        
        # for name, param in self.pretrained.named_parameters():
        #     if "lora" in name:
        #         param.requires_grad = True

        # 线性降维层
        self.fc = nn.Linear(dim, features)

    def forward(self, x):
        x = x.float()
        h, w = x.shape[-2:]
        max_size = 350  # 限制最大尺寸
        h = min(h, max_size)
        w = min(w, max_size)
        # 允许 `pretrained` 进行微调
        depth_features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        # with torch.no_grad():
        patch_h, patch_w = h // 14, w // 14
        # 冻结 `depth_head` 计算
        depth = self.depth_head(depth_features, patch_h, patch_w)
        # 这里不确定用(240, 320)还是(h, w)
        depth = F.interpolate(depth, size=(240, 320), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = -depth

        # 仅 `pretrained` 计算梯度
        depth_features = [pair[0] for pair in depth_features]
        features = torch.stack(depth_features, dim=1)  # [batch_size, 4, 625, 384]
        features = features.mean(dim=2)  # [batch_size, 4, 384]
        features = self.fc(features)  # [batch_size, 4, 256]
        features = features.mean(dim=1)  # [batch_size, 256]

        # rgb_data = x[0,:,:,:]
        # image = np.transpose(rgb_data.cpu().numpy().astype(np.uint8)*255, (1, 2, 0))  # 变换为 (H, W, C)
        # print("image_shape:",image.shape)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(image) 
        # plt.colorbar(label="rgb Value")  # 显示颜色条
        # plt.axis("off")

        # # 保存为 PNG 文件
        # plt.savefig("./rgb.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        # plt.close()

        # batch_depth_sample = depth.squeeze(1)[0,:,:]
        # batch_depth_sample = (batch_depth_sample).cpu().numpy()
        # vmin, vmax = np.percentile(batch_depth_sample, [5, 95])  # 去掉极端值以提升可视化效果
        # # image = Image.fromarray(batch_depth_sample, mode="L")
        # # image.save("depth_image.png")
        # # 可视化并保存灰度深度图
        # plt.figure(figsize=(8, 6))
        # plt.imshow(batch_depth_sample, cmap='viridis')  # 选择合适的颜色映射
        # plt.colorbar(label="Depth Value")  # 显示颜色条
        # plt.axis("off")
        # # 保存为 PNG 文件
        # plt.savefig("./batch_depth_in_model.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        # plt.close()
        depth = depth.float()
        features = features.float()
        return depth.squeeze(1), features

class DepthAnything(DPT_DINOv2, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    