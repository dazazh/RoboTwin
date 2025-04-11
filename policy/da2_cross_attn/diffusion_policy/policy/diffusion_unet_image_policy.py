from typing import Dict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs = self.normalizer.unnormalize(nobs)
        value = next(iter(nobs.values()))
        
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def affine_normalize(self, batch_depth):
        """
        对 batch 维度的 GT 深度进行仿射不变归一化（逐样本计算，但优化运算效率）
        :param batch_depth: 原始深度图 (Tensor) [B, H, W]
        :return: 归一化后的深度图 (Tensor) [B, H, W]
        """
        B, H, W = batch_depth.shape  # 获取 B帧(128*3) 维度信息

        # 计算中位数 t(d)（对每张图像单独计算）
        t_d = torch.median(batch_depth.view(B, -1), dim=1, keepdim=True)[0]  # [B, 1]
        
        # 计算尺度因子 s(d)（对每张图像单独计算）
        s_d = torch.mean(torch.abs(batch_depth.view(B, -1) - t_d), dim=1, keepdim=True)  # [B, 1]

        # 维度对齐，方便广播
        t_d = t_d.view(B, 1, 1)  # [B, 1, 1]
        s_d = s_d.view(B, 1, 1)  # [B, 1, 1]

        # 归一化
        normalized_depth = (batch_depth - t_d) / s_d  # [B, H, W]

        return normalized_depth

    def compute_loss(self, batch, epoch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        # 是否需要取消线性层
        nobs = self.normalizer.unnormalize(nobs)
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        # print("this_target_batch_depth:",this_target_batch_depth.shape)
        # print("this_nobs_rgb:",this_nobs['head_cam'].shape)
        # print("batch_depth:",batch_depth.shape)

        # 可视化输入rgb是否有问题+验证normalizer是否学到正确参数
        # rgb_data = nobs['head_cam'][0,0,:,:,:]
        # print("in_compute_loss_rgb_shape:",rgb_data.shape)
        # image = np.transpose(rgb_data.cpu().numpy(), (1, 2, 0))  # 变换为 (H, W, C)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(image) 
        # plt.colorbar(label="rgb Value")  # 显示颜色条
        # plt.axis("off")
        # plt.savefig("./in_compute_loss_rgb_with_linear_normalizer.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        # plt.close()

        # 检查depth anything的gt和模型输出
        # batch_depth_sample = batch_depth[0,:,:]
        # batch_depth_sample = (batch_depth_sample).detach().cpu().numpy()
        # vmin, vmax = np.percentile(batch_depth_sample, [5, 95])  # 去掉极端值以提升可视化效果
        # plt.figure(figsize=(8, 6))
        # plt.imshow(batch_depth_sample, cmap='viridis')  # 选择合适的颜色映射
        # plt.colorbar(label="Depth Value")  # 显示颜色条
        # plt.axis("off")
        # plt.savefig("./test_depth/batch_depth.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        # plt.close()

        # target_depth_sample = this_target_batch_depth[0,:,:]
        # target_depth_sample = (target_depth_sample).detach().cpu().numpy()
        # vmin, vmax = np.percentile(target_depth_sample, [5, 95])  # 去掉极端值以提升可视化效果
        # plt.figure(figsize=(8, 6))
        # plt.imshow(target_depth_sample, cmap='viridis')  # 选择合适的颜色映射
        # plt.colorbar(label="Depth Value")  # 显示颜色条
        # plt.axis("off")
        # # 保存为 PNG 文件
        # plt.savefig("./test_depth/target_depth.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        # plt.close()
        # assert False
        
        # depth_loss = F.mse_loss(batch_depth, this_target_batch_depth, reduction='none')
        # depth_loss = reduce(depth_loss, 'b ... -> b (...)', 'mean')
        
        loss = loss.mean()
        # depth_loss = depth_loss.mean()

        # initial_lambda = 0.1
        # decay_factor = 0.99  # 逐渐降低 depth loss 影响
        # lambda_depth = initial_lambda * (decay_factor ** epoch)

        # # 归一化 depth loss
        # depth_loss = depth_loss / (depth_loss.detach().mean() + 1e-6)

        # # 计算总损失
        # tot_loss = loss + lambda_depth * depth_loss
        # # print("depth_loss:",depth_loss)
        # tot_loss = loss + self.lambda_depth * depth_loss

        # print("loss:",loss)
        # print("depth_loss:",depth_loss)
        return loss
