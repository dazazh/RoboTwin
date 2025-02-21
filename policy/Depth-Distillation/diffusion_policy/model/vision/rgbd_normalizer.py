import torch
import torchvision
from torch import nn

class RGBDNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(RGBDNormalizer, self).__init__()
        self.mean = mean
        self.std = std
        self.rgb_normalizer = torchvision.transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        # print("sample:",sample.shape)
        # 假设sample的形状为[1, 4, H, W] (batch size = 1, 4 channels: RGB + Depth)
        
        # 提取RGB部分和深度部分
        rgb = sample[:, :3, :, :]  # RGB channels (first 3 channels)
        depth = torch.unsqueeze(sample[:, 3, :, :],dim=1)  # Depth channel (last channel)

        # Normalize RGB channels (单独对RGB进行归一化)
        rgb = self.rgb_normalizer(rgb.squeeze(0))  # 去掉batch维度来适应Normalize

        # 把归一化后的RGB与深度通道重新拼接
        # 注意: 需要将rgb和depth维度扩展回来以适应输出格式
        if sample.shape[0] == 1:
            rgb = rgb.unsqueeze(0)  # 重新加回批量维度
        # print("before:{},{}".format(rgb.shape,depth.shape))
        # print("after:",torch.cat((rgb, depth), dim=1).shape)
        return torch.cat((rgb, depth), dim=1)  # 拼接，dim=1表示拼接通道维度
