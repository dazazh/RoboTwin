import torch
import pickle
import os
from vggt.models.vggt import VGGT
import numpy as np
from PIL import Image
from torchvision import transforms as TF
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

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

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images.unsqueeze(0)

def visualize_feature_maps(features, output_path):
    """
    可视化特征图并保存
    
    Args:
        features (numpy.ndarray): 特征图数组，形状为 [n_features, height, width]
        output_path (str): 输出文件路径
    """
    n_features = features.shape[0]
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.ravel()
    
    for idx in range(n_features):
        feature_map = features[idx]
        # 归一化到0-1范围以便可视化
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        axes[idx].imshow(feature_map, cmap='viridis')
        axes[idx].set_title(f'Feature Map {idx*16}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存特征图可视化结果
    vis_output_path = output_path.replace('.pkl', '_features.png')
    plt.savefig(vis_output_path)
    plt.close()

def process_pkl_with_vggt(pkl_path, output_path, model, should_visualize):
    # 加载pkl文件
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 获取图像数据
    head_cam = data['observation']['head_camera']['rgb']
    front_cam = data['observation']['front_camera']['rgb']
    
    device = next(model.parameters()).device
    images = load_and_preprocess_images([head_cam, front_cam])
    images = images.to(device)
    
    # 获取VGGT的中间层特征
    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = model.aggregator(images)
        vggt_features = model.point_head(
            aggregated_tokens_list, 
            images=images, 
            patch_start_idx=patch_start_idx
        )
    
    # 将256个通道每16个连续通道进行平均池化，得到16个通道
    vggt_features = vggt_features.squeeze(0)  # 去掉batch维度
    vggt_features = vggt_features.view(16, 16, vggt_features.shape[-2], vggt_features.shape[-1])  # 重组为(16组, 每组16通道, H, W)
    selected_features = vggt_features.mean(dim=1)  # 在每组内进行平均池化
    data['vggt_features'] = selected_features.cpu().numpy()
    print("Pooled vggt_features shape: ", data['vggt_features'].shape)
    
    # 可视化特征图
    features = selected_features.cpu().numpy()  # 取第一个batch的特征图
    if should_visualize == 1:
        visualize_feature_maps(features, output_path)
    
    # 保存更新后的数据
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Process robot data with VGGT embeddings')
    parser.add_argument('--task', type=str, default='transparent_cup_place',
                      help='Task name (e.g., transparent_cup_place)')
    parser.add_argument('--should_visualize', type=int, default=0,
                      help='Whether to visualize feature maps')
    args = parser.parse_args()

    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT().to(device)
    ckpt = torch.load('/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/policy/dp_vggt/vggt/checkpoints/original_model.pt')
    model.load_state_dict(ckpt, strict=False)
    
    # 设置输入输出路径
    base_input_dir = f"/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/data/{args.task}_L515_pkl"
    base_output_dir = f"/cpfs04/shared/muyao/yuhao/data/{args.task}_L515_with_embedding_pkl"
    
    print(f"Processing task: {args.task}")
    print(f"Input directory: {base_input_dir}")
    print(f"Output directory: {base_output_dir}")
    
    # 遍历所有episode文件夹
    episode_dirs = [d for d in os.listdir(base_input_dir) if d.startswith('episode')]
    for episode_dir in tqdm(sorted(episode_dirs), desc="Processing episodes"):
        input_episode_dir = os.path.join(base_input_dir, episode_dir)
        output_episode_dir = os.path.join(base_output_dir, episode_dir)
        os.makedirs(output_episode_dir, exist_ok=True)
        
        # 处理当前episode下的所有pkl文件
        pkl_files = [f for f in sorted(os.listdir(input_episode_dir)) if f.endswith('.pkl')]
        for filename in tqdm(pkl_files, desc=f"Processing {episode_dir}", leave=False):
            input_path = os.path.join(input_episode_dir, filename)
            output_path = os.path.join(output_episode_dir, filename)
            process_pkl_with_vggt(input_path, output_path, model, args.should_visualize)

if __name__ == "__main__":
    main()