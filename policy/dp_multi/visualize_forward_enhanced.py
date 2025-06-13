import sys
sys.path.append('../..')
sys.path.insert(0, '../Multi-Diffusion-Policy')

import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # 移除seaborn依赖
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

def visualize_multiple_samples(obs_encoder, dataset, n_samples=4, save_dir="visualizations"):
    """可视化多个样本的处理过程"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] 开始可视化 {n_samples} 个样本...")
    
    n_rgb_keys = len(obs_encoder.rgb_keys)
    if n_rgb_keys == 0:
        print("[WARNING] 没有RGB观察键")
        return
    
    # 创建大图：样本 x RGB键 x (原始/变换后)
    fig, axes = plt.subplots(n_samples * 2, n_rgb_keys, figsize=(5*n_rgb_keys, 4*n_samples))
    if n_rgb_keys == 1:
        axes = axes.reshape(-1, 1)
    
    all_features = []
    
    for sample_idx in range(n_samples):
        print(f"[INFO] 处理样本 {sample_idx}...")
        
        # 获取样本数据
        sample = dataset[sample_idx * 10]  # 间隔取样
        batch = dataset.postprocess(sample, 'cuda:0')
        
        obs_dict = {}
        for key, value in batch['obs'].items():
            obs_dict[key] = value
        
        # 处理每个RGB键
        for rgb_idx, key in enumerate(obs_encoder.rgb_keys):
            img = obs_dict[key]
            
            # 原始图像
            original_img = img[0].detach().cpu().numpy()
            if original_img.ndim == 4:
                original_img = original_img[-1]  # 取最后一帧
            original_img = np.transpose(original_img, (1, 2, 0))
            original_img = np.clip(original_img, 0, 1)
            
            row_orig = sample_idx * 2
            axes[row_orig, rgb_idx].imshow(original_img)
            axes[row_orig, rgb_idx].set_title(f'样本{sample_idx} - {key} 原始')
            axes[row_orig, rgb_idx].axis('off')
            
            # 变换后图像
            img_transformed = obs_encoder.key_transform_map[key](img)
            transformed_img = img_transformed[0].detach().cpu().numpy()
            if transformed_img.ndim == 4:
                transformed_img = transformed_img[-1]
            transformed_img = np.transpose(transformed_img, (1, 2, 0))
            transformed_img = np.clip(transformed_img, 0, 1)
            
            row_trans = sample_idx * 2 + 1
            axes[row_trans, rgb_idx].imshow(transformed_img)
            axes[row_trans, rgb_idx].set_title(f'样本{sample_idx} - {key} 变换后')
            axes[row_trans, rgb_idx].axis('off')
        
        # 获取完整特征
        with torch.no_grad():
            features = obs_encoder(obs_dict)
            all_features.append(features.cpu().numpy())
    
    plt.tight_layout()
    plt.savefig(save_dir / 'multi_sample_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[INFO] 多样本对比图已保存到: {save_dir / 'multi_sample_comparison.png'}")
    plt.close()
    
    return np.array(all_features)

def analyze_feature_distribution(features, save_dir="visualizations"):
    """分析特征分布"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] 分析特征分布...")
    print(f"  - 特征形状: {features.shape}")
    
    # 特征统计
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 每个样本的特征分布
    axes[0, 0].boxplot([features[i].flatten() for i in range(features.shape[0])],
                       labels=[f'样本{i}' for i in range(features.shape[0])])
    axes[0, 0].set_title('各样本特征分布')
    axes[0, 0].set_ylabel('特征值')
    
    # 2. 特征维度的方差
    feature_vars = np.var(features, axis=0)
    axes[0, 1].hist(feature_vars, bins=50, alpha=0.7)
    axes[0, 1].set_title('特征维度方差分布')
    axes[0, 1].set_xlabel('方差')
    axes[0, 1].set_ylabel('频次')
    
    # 3. 平均特征值热图
    mean_features = np.mean(features, axis=0)
    if len(mean_features) > 100:
        # 如果特征太多，重塑为2D
        side_len = int(np.sqrt(len(mean_features)))
        if side_len * side_len <= len(mean_features):
            reshaped_features = mean_features[:side_len*side_len].reshape(side_len, side_len)
            im = axes[1, 0].imshow(reshaped_features, cmap='viridis')
            axes[1, 0].set_title('平均特征热图')
            plt.colorbar(im, ax=axes[1, 0])
        else:
            axes[1, 0].plot(mean_features)
            axes[1, 0].set_title('平均特征值曲线')
    else:
        axes[1, 0].bar(range(len(mean_features)), mean_features)
        axes[1, 0].set_title('平均特征值')
    
    # 4. 样本间相似性矩阵
    similarity_matrix = np.corrcoef(features.reshape(features.shape[0], -1))
    im = axes[1, 1].imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('样本间相似性矩阵')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_analysis.png', dpi=150, bbox_inches='tight')
    print(f"[INFO] 特征分析图已保存到: {save_dir / 'feature_analysis.png'}")
    plt.close()

def visualize_attention_maps(obs_encoder, obs_dict, save_dir="visualizations"):
    """可视化注意力图（如果模型支持）"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"[INFO] 尝试可视化注意力图...")
    
    # 注册hook来捕获中间激活
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 为每个RGB模型注册hook
    hooks = []
    for key in obs_encoder.rgb_keys:
        if key in obs_encoder.key_model_map:
            model = obs_encoder.key_model_map[key]
            # 尝试找到卷积层
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    hook = module.register_forward_hook(hook_fn(f"{key}_{name}"))
                    hooks.append(hook)
    
    # 前向传播
    with torch.no_grad():
        output = obs_encoder(obs_dict)
    
    # 可视化激活
    if activations:
        n_activations = min(len(activations), 4)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, (name, activation) in enumerate(list(activations.items())[:n_activations]):
            # 取第一个batch和时间步
            act = activation[0]
            if act.dim() == 4:  # (T, C, H, W)
                act = act[-1]   # 取最后一个时间步
            if act.dim() == 3:  # (C, H, W)
                # 平均所有通道
                act_map = torch.mean(act, dim=0).cpu().numpy()
                im = axes[idx].imshow(act_map, cmap='hot')
                axes[idx].set_title(f'{name}')
                axes[idx].axis('off')
                plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_dir / 'activation_maps.png', dpi=150, bbox_inches='tight')
        print(f"[INFO] 激活图已保存到: {save_dir / 'activation_maps.png'}")
        plt.close()
    
    # 清理hooks
    for hook in hooks:
        hook.remove()

def create_comprehensive_report(policy, dataset, save_dir="visualizations"):
    """创建综合分析报告"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    obs_encoder = policy.obs_encoder
    
    print(f"[INFO] 创建综合分析报告...")
    
    # 1. 多样本可视化
    features = visualize_multiple_samples(obs_encoder, dataset, n_samples=6, save_dir=save_dir)
    
    # 2. 特征分布分析
    analyze_feature_distribution(features, save_dir=save_dir)
    
    # 3. 单个样本详细分析
    sample = dataset[0]
    batch = dataset.postprocess(sample, 'cuda:0')
    obs_dict = {key: value for key, value in batch['obs'].items()}
    
    # 4. 注意力图可视化
    visualize_attention_maps(obs_encoder, obs_dict, save_dir=save_dir)
    
    # 5. 生成文本报告
    with open(save_dir / "comprehensive_report.txt", "w", encoding='utf-8') as f:
        f.write("MultiImageObsEncoder 综合分析报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 模型配置信息\n")
        f.write("-"*20 + "\n")
        f.write(f"RGB键: {obs_encoder.rgb_keys}\n")
        f.write(f"低维键: {obs_encoder.low_dim_keys}\n")
        f.write(f"共享RGB模型: {obs_encoder.share_rgb_model}\n")
        f.write(f"形状映射: {obs_encoder.key_shape_map}\n\n")
        
        f.write("2. 特征统计信息\n")
        f.write("-"*20 + "\n")
        f.write(f"特征形状: {features.shape}\n")
        f.write(f"特征范围: [{np.min(features):.6f}, {np.max(features):.6f}]\n")
        f.write(f"特征均值: {np.mean(features):.6f}\n")
        f.write(f"特征标准差: {np.std(features):.6f}\n\n")
        
        f.write("3. 维度分析\n")
        f.write("-"*20 + "\n")
        feature_vars = np.var(features, axis=0)
        f.write(f"高方差特征维度数量 (>0.1): {np.sum(feature_vars > 0.1)}\n")
        f.write(f"低方差特征维度数量 (<0.01): {np.sum(feature_vars < 0.01)}\n")
        f.write(f"最大方差: {np.max(feature_vars):.6f}\n")
        f.write(f"最小方差: {np.min(feature_vars):.6f}\n\n")
        
        f.write("4. 生成的可视化文件\n")
        f.write("-"*20 + "\n")
        f.write("- multi_sample_comparison.png: 多样本对比图\n")
        f.write("- feature_analysis.png: 特征分布分析图\n")
        f.write("- activation_maps.png: 激活图（如果可用）\n")
        f.write("- comprehensive_report.txt: 本报告\n")
    
    print(f"[INFO] 综合报告已保存到: {save_dir / 'comprehensive_report.txt'}")

def main():
    """主函数"""
    print("="*60)
    print("MultiImageObsEncoder 增强可视化工具")
    print("="*60)
    
    try:
        # 加载策略和数据集
        policy, dataset, cfg = load_policy_and_dataset()
        
        # 创建综合分析报告
        create_comprehensive_report(policy, dataset)
        
        print(f"\n[SUCCESS] 所有可视化完成!")
        print(f"[INFO] 请查看 visualizations/ 目录下的文件")
        
    except Exception as e:
        print(f"[ERROR] 可视化失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 