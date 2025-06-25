from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import matplotlib.pyplot as plt
from .model_getter import get_resnet, check_model_frozen
from .fast_vggt import VGGTPredictor as FastVGGTModel
from .vggt_adapter import VGGTAdapter
from vggt.models.vggt import VGGT
import cv2
import numpy as np

target_size = 518
patch_size = 14

def get_target_shape(width, height, mode="crop"):
    """
    Calculate target shape based on input dimensions and mode.
    
    Args:
        width (int): Original image width
        height (int): Original image height
        mode (str): Either "crop" or "pad"
        
    Returns:
        tuple: (new_height, new_width)
    """
    if mode == "pad":
        # Make the largest dimension 518px while maintaining aspect ratio
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size
    else:  # mode == "crop"
        # Set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by patch_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size
        
        # Center crop height if it's larger than target_size
        if new_height > target_size:
            new_height = target_size
            
    return new_height, new_width

def preprocess_rgb(rgb, mode="crop"):
    """
    Preprocess RGB image with the following steps:
    1. Normalize to [0, 1]
    2. Resize to target shape (maintaining aspect ratio and divisible by patch_size)
    3. Center crop or pad if necessary
    4. Transpose to (C, H, W)
    
    Args:
        rgb (numpy.ndarray): Input RGB image with shape (H, W, C)
        mode (str): Either "crop" or "pad"
        
    Returns:
        numpy.ndarray: Preprocessed image with shape (C, H, W)
    """
    # Convert to float32 and normalize to [0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    
    # Get original dimensions
    height, width = rgb.shape[:2]
    
    # Calculate target dimensions
    new_height, new_width = get_target_shape(width, height, mode)
    
    # Resize image
    rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Handle cropping or padding
    if mode == "crop":
        # Center crop if height is larger than target_size
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            rgb = rgb[start_y:start_y + target_size, :, :]
    else:  # mode == "pad"
        # Pad to make a square of target_size x target_size
        h_padding = target_size - rgb.shape[0]
        w_padding = target_size - rgb.shape[1]
        
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            
            # Pad with white (value=1.0)
            rgb = np.pad(rgb, 
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='constant',
                        constant_values=1.0)
    
    # Transpose to (C, H, W)
    rgb = np.transpose(rgb, (2, 0, 1))
    
    return rgb

def extract_features(vggt_model, img_tensor, device):
    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = vggt_model(img_tensor.to(device))
        intermediate_layer_idx = [4, 11, 17, 23]
        intermediate_tokens = [aggregated_tokens_list[intermediate_layer_idx[i]][:,:,patch_start_idx:] for i in range(len(intermediate_layer_idx))]
        intermediate_features = torch.cat(intermediate_tokens, dim=1)
        intermediate_features = intermediate_features.mean(dim=2)

    return intermediate_features

def load_vggt_model():
    model = VGGT()
    model.load_state_dict(torch.load("/data/user/xcs/yuhao/3d-aware/fast_vggt/vggt/checkpoint/original_model.pt"))
    return model

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
            # spatial_reducer: nn.Module=None 
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
            elif type == 'original_rgb':
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        # rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        feature_keys = sorted(feature_keys)

        self.fast_vggt_model = FastVGGTModel()
        fast_vggt_ckpt = torch.load('/data/user/xcs/yuhao/3d-aware/fast_vggt/checkpoints/normalized/vggt_predictor_epoch100.pth')
        self.fast_vggt_model.load_state_dict(fast_vggt_ckpt, strict=False)
        self.fast_vggt_model = self.fast_vggt_model.to(self.device)
        self.vggt_adapter = VGGTAdapter(output_dim=512).to(self.device)
        self.vggt_model = load_vggt_model().to(self.device)
        self.step = 0
        self.current_vggt_features = None

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.feature_keys = feature_keys
        self.key_shape_map = key_shape_map
        # self.spatial_reducer = spatial_reducer
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

    def forward(self, obs_dict, original_obs_dict=None):
        if original_obs_dict is None:
            return self._forward_train(obs_dict)
        else:
            return self._forward_inference(obs_dict, original_obs_dict)

    def _forward_train(self, obs_dict):
        batch_size = None
        features = list()

        # run each rgb obs to independent models
        batch_fast_vggt = {}
        for key in self.rgb_keys:
            img = obs_dict[key]
            # 由于postprocess删掉了原来的归一化，所以在这里需要重新归一化
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == torch.Size(self.key_shape_map[key])
            # fast_vggt_img = torch.randn_like(fast_vggt_img)

            img = self.key_transform_map[key](img)
            feature = self.key_model_map[key](img)

            features.append(feature)
            if key == 'head_cam':
                batch_fast_vggt['head_img'] = obs_dict['vggt_head_cam']
            elif key == 'front_cam':
                batch_fast_vggt['front_img'] = obs_dict['vggt_front_cam']
        
        # 确保所有张量都在正确的设备上
        batch_fast_vggt['vggt_feat_t0'] = obs_dict['vggt_features']
        vggt_features_t1_gt = obs_dict['vggt_features_current']

        vggt_features_t1 = self.fast_vggt_model(batch_fast_vggt)
        # print("vggt_features_t0", batch_fast_vggt['vggt_feat_t0'])
        # print("vggt_features_t1", vggt_features_t1)
        # print("vggt_features_t1_gt", vggt_features_t1_gt)
        mse_loss = torch.nn.functional.mse_loss(vggt_features_t1, vggt_features_t1_gt)
        # print("mse loss:", mse_loss)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # 添加可视化
        # if self.training:  # 只在训练时可视化
        #     self.visualize_vggt_features(
        #         vggt_features, 
        #         f'vggt_features_visualization_{self.visualization_counter}.png'
        #     )
        #     self.visualization_counter += 1
        vggt_features = self.vggt_adapter(vggt_features_t1)
        features.append(vggt_features)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result, mse_loss
        

    def _forward_inference(self, obs_dict, vggt_obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        vggt_head_img = vggt_obs_dict['head_cam']
        vggt_front_img = vggt_obs_dict['front_cam']
        vggt_imgs = [vggt_head_img, vggt_front_img]

        batch_fast_vggt = {}
        batch_fast_vggt['head_img'] = vggt_head_img.to(dtype=torch.float32)
        batch_fast_vggt['front_img'] = vggt_front_img.to(dtype=torch.float32)

        for key in self.rgb_keys:
            img = obs_dict[key]
            vggt_img = vggt_obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == torch.Size(self.key_shape_map[key])
            img = self.key_transform_map[key](img)
            feature = self.key_model_map[key](img)
            features.append(feature)
            # vggt_imgs.append(vggt_img)
            # if key == 'head_cam':
            #     batch_fast_vggt['head_img'] = obs_dict['vggt_head_cam'].to(dtype=torch.float32)
            # elif key == 'front_cam':
            #     batch_fast_vggt['front_img'] = obs_dict['vggt_front_cam'].to(dtype=torch.float32)
        
        # 确保所有张量都在正确的设备上
        if self.step % 5 == 0:
            vggt_imgs = torch.stack(vggt_imgs, dim=1)
            vggt_features = extract_features(self.vggt_model, vggt_imgs, self.device)
            self.current_vggt_features = vggt_features
        else:
            batch_fast_vggt['vggt_feat_t0'] = self.current_vggt_features.to(dtype=torch.float32)
            batch_fast_vggt = batch_fast_vggt
            vggt_features = self.fast_vggt_model(batch_fast_vggt)
            self.current_vggt_features = vggt_features
        vggt_features = self.vggt_adapter(vggt_features)

        # 添加可视化
        # if self.training:  # 只在训练时可视化
        #     self.visualize_vggt_features(
        #         vggt_features, 
        #         f'vggt_features_visualization_{self.visualization_counter}.png'
        #     )
        #     self.visualization_counter += 1
        features.append(vggt_features)
        
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
        self.step += 1
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
        example_output, _ = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
