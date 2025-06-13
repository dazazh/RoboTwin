import sys
sys.path.insert(0, '.')  # 优先使用当前目录的模块
sys.path.append('../..')
# sys.path.insert(0, '../Multi-Diffusion-Policy')  # 注释掉这行，避免导入错误的版本

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
import dill

# 导入必要的模块 - 确保使用本地版本
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

def quick_test():
    """快速测试MultiImageObsEncoder的forward函数"""
    print("🚀 快速测试 MultiImageObsEncoder Forward 函数")
    print("="*50)
    
    try:
        # 1. 加载模型
        print("📦 加载模型...")
        checkpoint_path = 'checkpoints/transparent_cup_place_L515_300_0/300.ckpt'
        
        # # 检查checkpoint是否存在
        # if not Path(checkpoint_path).exists():
        #     print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        #     # 尝试查找其他可用的checkpoint
        #     import os
        #     for root, dirs, files in os.walk('checkpoints'):
        #         for file in files:
        #             if file.endswith('.ckpt'):
        #                 checkpoint_path = os.path.join(root, file)
        #                 print(f"🔄 使用找到的checkpoint: {checkpoint_path}")
        #                 break
        #         if checkpoint_path != 'checkpoints/transparent_cup_place_L515_300_0/300.ckpt':
        #             break
        
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to('cuda:0')
        policy.eval()
        
        # 2. 加载一个样本数据
        print("📊 加载测试数据...")
        
        # 尝试多个可能的zarr路径
        possible_zarr_paths = [
            "data/transparent_cup_place_L515_50.zarr",
            "../data/transparent_cup_place_L515_300.zarr", 
            "../../data/transparent_cup_place_L515_300.zarr",
            "../../../data/transparent_cup_place_L515_300.zarr"
        ]
        
        zarr_path = None
        for path in possible_zarr_paths:
            if Path(path).exists():
                zarr_path = path
                print(f"✅ 找到数据文件: {zarr_path}")
                break
        
        if zarr_path is None:
            print("❌ 找不到zarr数据文件，尝试的路径:")
            for path in possible_zarr_paths:
                print(f"   - {path} ({'存在' if Path(path).exists() else '不存在'})")
            return
        
        dataset = RobotImageDataset(
            zarr_path=zarr_path,
            horizon=cfg.horizon,
            pad_before=cfg.n_obs_steps-1,
            pad_after=cfg.n_action_steps-1,
            seed=42,
            val_ratio=0.02
        )
        
        sample = dataset[0]
        
        # 调试信息
        print(f"🔍 调试信息:")
        print(f"   - Dataset类型: {type(dataset)}")
        print(f"   - Sample键: {list(sample.keys())}")
        print(f"   - Postprocess方法: {dataset.postprocess}")
        
        batch = dataset.postprocess(sample, 'cuda:0')
        obs_dict = batch['obs']
        
        print(f"✅ 数据加载完成")
        print(f"   - 观察键: {list(obs_dict.keys())}")
        for key, value in obs_dict.items():
            print(f"   - {key}: {value.shape}")
        
        # 3. 测试forward函数
        print("\n🔍 测试 MultiImageObsEncoder.forward()...")
        obs_encoder = policy.obs_encoder
        
        print(f"   - RGB键: {obs_encoder.rgb_keys}")
        print(f"   - 低维键: {obs_encoder.low_dim_keys}")
        print(f"   - 共享RGB模型: {obs_encoder.share_rgb_model}")
        
        with torch.no_grad():
            # 调用forward函数
            features = obs_encoder(obs_dict)
            print(f"✅ Forward函数执行成功!")
            print(f"   - 输出特征形状: {features.shape}")
            print(f"   - 特征范围: [{torch.min(features):.4f}, {torch.max(features):.4f}]")
            print(f"   - 特征均值: {torch.mean(features):.4f}")
        
        # 4. 简单可视化
        print("\n🎨 生成简单可视化...")
        Path("visualizations").mkdir(exist_ok=True)
        
        # 可视化输入图像
        fig, axes = plt.subplots(1, len(obs_encoder.rgb_keys), figsize=(5*len(obs_encoder.rgb_keys), 5))
        if len(obs_encoder.rgb_keys) == 1:
            axes = [axes]
        
        for i, key in enumerate(obs_encoder.rgb_keys):
            img = obs_dict[key][0, -1].detach().cpu().numpy()  # 取最后一帧
            
            print(f"🖼️  {key} 图像形状: {img.shape}")
            
            # 处理不同的图像格式
            if img.ndim == 3:  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # 转换为 (H, W, C)
            elif img.ndim == 4:  # (T, C, H, W) - 不应该发生，但以防万一
                img = img[-1]  # 取最后一帧
                img = np.transpose(img, (1, 2, 0))
            elif img.ndim == 2:  # 灰度图 (H, W)
                pass  # 保持原样
            else:
                print(f"⚠️  未知的图像维度: {img.shape}")
                continue
                
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            axes[i].set_title(f'{key}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/quick_test_images.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 可视化特征
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.plot(features[0].cpu().numpy())
        plt.title('特征值曲线')
        plt.xlabel('特征维度')
        plt.ylabel('值')
        
        plt.subplot(132)
        plt.hist(features[0].cpu().numpy(), bins=50, alpha=0.7)
        plt.title('特征值分布')
        plt.xlabel('值')
        plt.ylabel('频次')
        
        plt.subplot(133)
        # 如果特征维度足够大，显示为2D热图
        feat_np = features[0].cpu().numpy()
        if len(feat_np) >= 64:
            side = int(np.sqrt(len(feat_np)))
            if side * side <= len(feat_np):
                feat_2d = feat_np[:side*side].reshape(side, side)
                plt.imshow(feat_2d, cmap='viridis')
                plt.title('特征热图')
                plt.colorbar()
            else:
                plt.plot(feat_np)
                plt.title('特征值')
        else:
            plt.bar(range(len(feat_np)), feat_np)
            plt.title('特征条形图')
        
        plt.tight_layout()
        plt.savefig('visualizations/quick_test_features.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ 可视化完成!")
        print("   - quick_test_images.png: 输入图像")
        print("   - quick_test_features.png: 输出特征")
        
        print(f"\n🎉 测试成功完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    quick_test() 