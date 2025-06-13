import sys
sys.path.append('../..')
sys.path.insert(0, '../Multi-Diffusion-Policy')

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import hydra
import dill
from omegaconf import OmegaConf

# 导入必要的模块
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

def load_policy_and_dataset(task_name='transparent_cup_place', head_camera_type='L515', 
                           expert_data_num=300, seed=0, checkpoint_num=300):
    """加载策略和数据集"""
    print(f"[INFO] 加载策略和数据集...")
    
    # 加载checkpoint
    checkpoint_path = f'checkpoints/{task_name}_{head_camera_type}_{expert_data_num}_{seed}/{checkpoint_num}.ckpt'
    
    # 检查checkpoint是否存在
    if not Path(checkpoint_path).exists():
        print(f"[WARNING] Checkpoint文件不存在: {checkpoint_path}")
        # 尝试查找其他可用的checkpoint
        for root, dirs, files in os.walk('checkpoints'):
            for file in files:
                if file.endswith('.ckpt'):
                    checkpoint_path = os.path.join(root, file)
                    print(f"[INFO] 使用找到的checkpoint: {checkpoint_path}")
                    break
            if checkpoint_path != f'checkpoints/{task_name}_{head_camera_type}_{expert_data_num}_{seed}/{checkpoint_num}.ckpt':
                break
    
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 创建workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # 获取策略
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to('cuda:0')
    policy.eval()
    
    # 加载数据集 - 尝试多个可能的路径
    possible_zarr_paths = [
        f"data/{task_name}_{head_camera_type}_50.zarr",
        f"../data/{task_name}_{head_camera_type}_{expert_data_num}.zarr",
        f"../../data/{task_name}_{head_camera_type}_{expert_data_num}.zarr",
        f"../../../data/{task_name}_{head_camera_type}_{expert_data_num}.zarr"
    ]
    
    zarr_path = None
    for path in possible_zarr_paths:
        if Path(path).exists():
            zarr_path = path
            print(f"[INFO] 找到数据文件: {zarr_path}")
            break
    
    if zarr_path is None:
        print("[ERROR] 找不到zarr数据文件，尝试的路径:")
        for path in possible_zarr_paths:
            print(f"   - {path} ({'存在' if Path(path).exists() else '不存在'})")
        raise FileNotFoundError("无法找到zarr数据文件")
    
    dataset = RobotImageDataset(
        zarr_path=zarr_path,
        horizon=cfg.horizon,
        pad_before=cfg.n_obs_steps-1,
        pad_after=cfg.n_action_steps-1,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=expert_data_num
    )
    
    print(f"[INFO] 策略和数据集加载完成")
    print(f"  - 策略类型: {type(policy)}")
    print(f"  - 观察编码器类型: {type(policy.obs_encoder)}")
    print(f"  - 数据集大小: {len(dataset)}")
    
    return policy, dataset, cfg

def visualize_images_in_forward(obs_encoder, obs_dict, save_dir="visualizations"):
    """在forward函数中可视化图像处理过程"""
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] 开始可视化forward函数中的图像处理...")
    print(f"[INFO] 输入观察字典键: {list(obs_dict.keys())}")
    
    batch_size = None
    features = list()
    
    # 创建图形
    n_rgb_keys = len(obs_encoder.rgb_keys)
    if n_rgb_keys > 0:
        fig, axes = plt.subplots(2, n_rgb_keys, figsize=(5*n_rgb_keys, 10))
        if n_rgb_keys == 1:
            axes = axes.reshape(-1, 1)
    
    # 处理RGB输入
    if obs_encoder.share_rgb_model:
        print("[INFO] 使用共享RGB模型...")
        imgs = list()
        for i, key in enumerate(obs_encoder.rgb_keys):
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            
            print(f"[INFO] 处理 {key}: shape={img.shape}, range=[{torch.min(img):.3f}, {torch.max(img):.3f}]")
            
            # 可视化原始图像 (第一个batch)
            original_img = img[0].detach().cpu().numpy()  # (T, C, H, W)
            if original_img.ndim == 4:  # 如果有时间维度，取最后一帧
                original_img = original_img[-1]  # (C, H, W)
            original_img = np.transpose(original_img, (1, 2, 0))  # (H, W, C)
            original_img = np.clip(original_img, 0, 1)
            
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'{key} - 原始图像')
            axes[0, i].axis('off')
            
            # 应用变换
            img_transformed = obs_encoder.key_transform_map[key](img)
            print(f"[INFO] {key} 变换后: shape={img_transformed.shape}, range=[{torch.min(img_transformed):.3f}, {torch.max(img_transformed):.3f}]")
            
            # 可视化变换后的图像
            transformed_img = img_transformed[0].detach().cpu().numpy()
            if transformed_img.ndim == 4:
                transformed_img = transformed_img[-1]
            transformed_img = np.transpose(transformed_img, (1, 2, 0))
            transformed_img = np.clip(transformed_img, 0, 1)
            
            axes[1, i].imshow(transformed_img)
            axes[1, i].set_title(f'{key} - 变换后')
            axes[1, i].axis('off')
            
            imgs.append(img_transformed)
        
        # 拼接所有图像
        imgs = torch.cat(imgs, dim=0)
        print(f"[INFO] 拼接后图像形状: {imgs.shape}")
        
        # 通过RGB模型
        feature = obs_encoder.key_model_map['rgb'](imgs)
        print(f"[INFO] RGB模型输出特征形状: {feature.shape}")
        
        # 重塑特征
        feature = feature.reshape(-1, batch_size, *feature.shape[1:])
        feature = torch.moveaxis(feature, 0, 1)
        feature = feature.reshape(batch_size, -1)
        features.append(feature)
        print(f"[INFO] 最终RGB特征形状: {feature.shape}")
        
    else:
        print("[INFO] 使用独立RGB模型...")
        for i, key in enumerate(obs_encoder.rgb_keys):
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            
            print(f"[INFO] 处理 {key}: shape={img.shape}, range=[{torch.min(img):.3f}, {torch.max(img):.3f}]")
            
            # 可视化原始图像
            original_img = img[0].detach().cpu().numpy()
            if original_img.ndim == 4:
                original_img = original_img[-1]
            original_img = np.transpose(original_img, (1, 2, 0))
            original_img = np.clip(original_img, 0, 1)
            
            axes[0, i].imshow(original_img)
            axes[0, i].set_title(f'{key} - 原始图像')
            axes[0, i].axis('off')
            
            # 应用变换
            img_transformed = obs_encoder.key_transform_map[key](img)
            print(f"[INFO] {key} 变换后: shape={img_transformed.shape}, range=[{torch.min(img_transformed):.3f}, {torch.max(img_transformed):.3f}]")
            
            # 可视化变换后的图像
            transformed_img = img_transformed[0].detach().cpu().numpy()
            if transformed_img.ndim == 4:
                transformed_img = transformed_img[-1]
            transformed_img = np.transpose(transformed_img, (1, 2, 0))
            transformed_img = np.clip(transformed_img, 0, 1)
            
            axes[1, i].imshow(transformed_img)
            axes[1, i].set_title(f'{key} - 变换后')
            axes[1, i].axis('off')
            
            # 通过独立模型
            feature = obs_encoder.key_model_map[key](img_transformed)
            features.append(feature)
            print(f"[INFO] {key} 特征形状: {feature.shape}")
    
    # 保存可视化图像
    if n_rgb_keys > 0:
        plt.tight_layout()
        plt.savefig(save_dir / 'rgb_processing.png', dpi=150, bbox_inches='tight')
        print(f"[INFO] RGB处理可视化已保存到: {save_dir / 'rgb_processing.png'}")
        plt.close()
    
    # 处理低维输入
    for key in obs_encoder.low_dim_keys:
        data = obs_dict[key]
        if batch_size is None:
            batch_size = data.shape[0]
        print(f"[INFO] 低维数据 {key}: shape={data.shape}, range=[{torch.min(data):.3f}, {torch.max(data):.3f}]")
        features.append(data)
    
    # 拼接所有特征
    result = torch.cat(features, dim=-1)
    print(f"[INFO] 最终输出特征形状: {result.shape}")
    
    return result

def create_sample_obs_dict(dataset, policy, sample_idx=0):
    """创建示例观察字典"""
    print(f"[INFO] 创建示例观察字典 (sample {sample_idx})...")
    
    # 从数据集获取样本
    sample = dataset[sample_idx]
    batch = dataset.postprocess(sample, 'cuda:0')
    
    # 提取观察数据
    obs_dict = {}
    for key, value in batch['obs'].items():
        # 确保数据格式正确 (B, T, ...)
        if value.dim() >= 2:
            obs_dict[key] = value  # 保持原始维度
        else:
            obs_dict[key] = value.unsqueeze(0)  # 添加batch维度
        print(f"[INFO] {key}: {obs_dict[key].shape}")
    
    return obs_dict

def main():
    """主函数"""
    print("="*50)
    print("MultiImageObsEncoder Forward 可视化工具")
    print("="*50)
    
    try:
        # 加载策略和数据集
        policy, dataset, cfg = load_policy_and_dataset()
        
        # 创建示例观察
        obs_dict = create_sample_obs_dict(dataset, policy, sample_idx=0)
        
        # 获取观察编码器
        obs_encoder = policy.obs_encoder
        print(f"[INFO] 观察编码器信息:")
        print(f"  - RGB键: {obs_encoder.rgb_keys}")
        print(f"  - 低维键: {obs_encoder.low_dim_keys}")
        print(f"  - 共享RGB模型: {obs_encoder.share_rgb_model}")
        
        # 可视化forward函数
        with torch.no_grad():
            output_features = visualize_images_in_forward(obs_encoder, obs_dict)
        
        print(f"[SUCCESS] 可视化完成!")
        print(f"[INFO] 输出特征形状: {output_features.shape}")
        print(f"[INFO] 特征范围: [{torch.min(output_features):.3f}, {torch.max(output_features):.3f}]")
        
        # 保存特征统计信息
        save_dir = Path("visualizations")
        with open(save_dir / "feature_stats.txt", "w") as f:
            f.write(f"输出特征统计信息\n")
            f.write(f"="*30 + "\n")
            f.write(f"特征形状: {output_features.shape}\n")
            f.write(f"特征范围: [{torch.min(output_features):.6f}, {torch.max(output_features):.6f}]\n")
            f.write(f"特征均值: {torch.mean(output_features):.6f}\n")
            f.write(f"特征标准差: {torch.std(output_features):.6f}\n")
        
        print(f"[INFO] 特征统计信息已保存到: {save_dir / 'feature_stats.txt'}")
        
    except Exception as e:
        print(f"[ERROR] 可视化失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 