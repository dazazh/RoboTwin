from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
# from diffusion_policy.model.vision.dino import DPT_DINOv2_Encoder
# from depth_anything.dpt import DepthAnything
# from depth_anything.dpt_encoder import DPT_DINOv2_Encoder
# from depth_anything.blocks import *
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms.v2 as transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from peft import PeftModel
from peft import LoraConfig, get_peft_model

def update_pretrained_keys(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # 去掉 'module.'
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            dino_encoder: nn.Module,
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # self.dino_encoder = dino_encoder.from_pretrained("LiheYoung/depth_anything_vits14")
        # self.dino_encoder = DPT_DINOv2_Encoder(encoder='vits', localhub=True).to(self.device)
        self.dino_encoder = dino_encoder
        # ckpt_path = "/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/policy/Diffusion-Policy-DA(loss)/depth_anything_v2/checkpoint/latest.pth"
        # checkpoint = update_pretrained_keys(torch.load(ckpt_path, map_location="cpu")['model'])
        # self.dino_encoder.load_state_dict(checkpoint, strict=False)  # `strict=False` 兼容部分加载
        # print(self.dino_encoder)
        # LoRA 适配 `qkv` 和 `proj` 层
        # print(self.dino_encoder)
        # self.dino_encoder.float()
        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        batch_depth = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            dino_feature = self.dino_encoder(imgs)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            dino_feature = dino_feature.reshape(-1,batch_size,*dino_feature.shape[1:])
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            dino_feature = torch.moveaxis(feature,0,1)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            dino_feature = dino_feature.reshape(batch_size,-1)
            feature = feature.reshape(batch_size,-1)
            feature = torch.cat((feature,dino_feature),dim=1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                dino_feature = self.dino_encoder(img)
                feature = self.key_model_map[key](img)
                feature = torch.cat((feature,dino_feature),dim=1)
                features.append(feature)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        # print(output_shape)
        return output_shape
