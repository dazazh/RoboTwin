from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            vggt_model: nn.Module=None,
            spatial_reducer: nn.Module=None 
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        Assumes features input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        feature_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

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
            elif type == 'features':
                feature_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        # rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        feature_keys = sorted(feature_keys)

        # Initialize vggt_model and ensure it follows device
        self.vggt_model = vggt_model
        # ckpt = torch.load('/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/policy/dp_vggt/vggt/checkpoints/original_model.pt')
        # self.vggt_model.load_state_dict(ckpt,strict=False)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.feature_keys = feature_keys
        self.key_shape_map = key_shape_map
        self.spatial_reducer = spatial_reducer
        self.visualization_counter = 0

    def visualize_vggt_features(self, features, output_path):
        """
        可视化VGGT特征图
        
        Args:
            features (torch.Tensor): 特征图，形状为 [n_features, height, width]
            output_path (str): 输出文件路径
        """
        # 确保特征图在CPU上并转换为numpy数组
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        # 如果特征图是4D的，取第一个batch
        if features.ndim == 4:
            features = features[0]
        
        n_features = features.shape[0]
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.ravel()
        
        for idx in range(n_features):
            feature_map = features[idx]
            # 归一化到0-1范围以便可视化
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            axes[idx].imshow(feature_map, cmap='viridis')
            axes[idx].set_title(f'Feature Map {idx}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def forward(self, obs_dict, original_obs_dict):
        batch_size = None
        features = list()
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
                assert img.shape[1:] == torch.Size(self.key_shape_map[key])
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            vggt_img = []
            # 只对第一个RGB key进行可视化
            first_key = self.rgb_keys[0]
            img = obs_dict[first_key]
            
            # 可视化第一张图片
            if img.dim() == 4:  # (B,C,H,W)
                img_to_vis = img[0].clone()  # 取第一个batch
                if isinstance(img_to_vis, torch.Tensor):
                    img_np = img_to_vis.detach().cpu().numpy()
                    if img_np.shape[0] <= 4:  # CHW -> HWC
                        img_np = np.transpose(img_np, (1, 2, 0))
                    
                    # 处理值范围
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
                    # 处理通道
                    if img_np.shape[2] == 1:
                        img_np = np.repeat(img_np, 3, axis=2)
                    elif img_np.shape[2] == 4:
                        img_np = img_np[:, :, :3]
                    
                    # 保存图像
                    pil_img = Image.fromarray(img_np)
                    pil_img.save('first_image.png')
                    print(f"第一张图像已保存为 first_image.png, 形状: {img.shape}")
            
            # 继续原来的处理流程
            for key in self.rgb_keys:
                img = obs_dict[key]
                original_img = original_obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == torch.Size(self.key_shape_map[key])
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
                vggt_img.append(original_img)
            
            # 确保所有张量都在正确的设备上
            vggt_img = torch.stack(vggt_img, dim=1).to(self.device)
            vggt_features = self.vggt_model(vggt_img)
            vggt_features = vggt_features.view(-1 ,16, 16, vggt_features.shape[-2], vggt_features.shape[-1])  # 重组为(16组, 每组16通道, H, W)
            vggt_features = vggt_features.mean(dim=2)  # 在每组内进行平均池化
            # if self.training:  # 只在训练时可视化
            # self.visualize_vggt_features(
            #     vggt_features[0], 
            #     f'vggt_features_visualization_{self.visualization_counter}.png'
            # )
            self.visualization_counter += 1
            reduced_vggt_features = self.spatial_reducer(vggt_features)
            features.append(reduced_vggt_features)
        
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
        example_vggt_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            vggt_obs = torch.zeros(
                (batch_size, 3, 518, 518), 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
            example_vggt_obs_dict[key] = vggt_obs
        example_output = self.forward(example_obs_dict,example_vggt_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
