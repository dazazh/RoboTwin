import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque

import pickle
import os
from vggt.models.vggt import VGGT
import numpy as np
from PIL import Image
from torchvision import transforms as TF
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

def load_and_preprocess_images(image_list):
    """
    A quick start function to preprocess numpy images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_list (list): List of numpy arrays representing images with shape (H, W, C)

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - The function ensures width=518px while maintaining aspect ratio
        - Height is adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for img in image_list:
        # Convert numpy array to PIL Image
        img = Image.fromarray(img)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        new_width = 518

        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = 518

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518
        if new_height > 518:
            start_y = (new_height - 518) // 2
            img = img[:, start_y : start_y + 518, :]

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images
    return images


class DPRunner:
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=300,
                 n_obs_steps=3,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
    ):
        self.task_name = task_name
        self.eval_episodes = eval_episodes
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.obs = deque(maxlen=n_obs_steps+1)
        self.env = None

    def stack_last_n_obs(self, all_obs, n_steps):
        assert(len(all_obs) > 0)
        all_obs = list(all_obs)
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.array(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros((n_steps,) + all_obs[-1].shape, 
                dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
        return result
    
    def reset_obs(self):
        self.obs.clear()

    def update_obs(self, current_obs):
        self.obs.append(current_obs)

    def get_n_steps_obs(self):
        assert(len(self.obs) > 0), 'no observation is recorded, please update obs first'

        result = dict()
        for key in self.obs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in self.obs],
                self.n_obs_steps
            )

        return result

    def get_action(self, policy: BaseImagePolicy, observaton=None):
        device, dtype = policy.device, policy.dtype
        if observaton is not None:
            self.obs.append(observaton) # update
        obs = self.get_n_steps_obs()

        # create obs dict
        np_obs_dict = dict(obs)
        # 分离VGGT相关的字段
        vggt_obs_dict = {}
        vggt_obs_dict['head_cam'] = np_obs_dict.pop('vggt_head_cam')
        vggt_obs_dict['front_cam'] = np_obs_dict.pop('vggt_front_cam')
        
        # device transfer
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        
        # 预处理VGGT图像
        if vggt_obs_dict:
            processed_vggt_obs = {'head_cam': [], 'front_cam': []}
            for i in range(3):
                # 处理每个时间步的图像
                processed_images = load_and_preprocess_images([
                    vggt_obs_dict['head_cam'][i], 
                    vggt_obs_dict['front_cam'][i]
                ])
                # 将处理后的图像分别存储
                processed_vggt_obs['head_cam'].append(processed_images[0])
                processed_vggt_obs['front_cam'].append(processed_images[1])
            # 将列表转换为张量
            processed_vggt_obs = {
                'head_cam': torch.stack(processed_vggt_obs['head_cam']),
                'front_cam': torch.stack(processed_vggt_obs['front_cam'])
            }
            vggt_obs_dict = processed_vggt_obs
        
        # run policy
        with torch.no_grad():
            obs_dict_input = {}  # flush unused keys
            obs_dict_input['head_cam'] = obs_dict['head_cam'].unsqueeze(0)
            obs_dict_input['front_cam'] = obs_dict['front_cam'].unsqueeze(0)
            obs_dict_input['left_cam'] = obs_dict['left_cam'].unsqueeze(0)
            obs_dict_input['right_cam'] = obs_dict['right_cam'].unsqueeze(0)
            obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
            vggt_obs_dict_input = {}
            vggt_obs_dict_input['head_cam'] = vggt_obs_dict['head_cam'].unsqueeze(0)
            vggt_obs_dict_input['front_cam'] = vggt_obs_dict['front_cam'].unsqueeze(0)
            
            action_dict = policy.predict_action(obs_dict_input,vggt_obs_dict_input)

        # device_transfer
        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)
        return action
