import os
import glob
import pickle
import torch
import zarr
import numpy as np
from tqdm import tqdm
import argparse
import cv2
import shutil
from vggt.models.vggt import VGGT
import matplotlib.pyplot as plt
# ======= 图片预处理 =======

target_size = 518
patch_size = 14
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_vggt_model():
    model = VGGT()
    model.load_state_dict(torch.load("/data/user/xcs/yuhao/3d-aware/fast_vggt/vggt/checkpoint/original_model.pt"))
    model = model.to(device)
    return model

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

# ======= VGGT特征提取 =======

def extract_features(vggt_model, img_tensor):
    with torch.no_grad():
        aggregated_tokens_list, patch_start_idx = vggt_model(img_tensor.to(device))
        intermediate_layer_idx = [4, 11, 17, 23]
        intermediate_tokens = [aggregated_tokens_list[intermediate_layer_idx[i]][:,:,patch_start_idx:] for i in range(len(intermediate_layer_idx))]
        intermediate_features = torch.cat(intermediate_tokens, dim=1)
        intermediate_features = intermediate_features.mean(dim=2)
    return intermediate_features.cpu().numpy()

# ======= 主流程 =======

def main():
    parser = argparse.ArgumentParser(description='pkl2zarr with vggt+action')
    parser.add_argument('task_name', type=str)
    parser.add_argument('camera_type', type=str)
    parser.add_argument('num_episodes', type=int)
    parser.add_argument('--pkl_root', type=str, default='/data/user/xcs/yuhao/3d-aware/RoboTwin/data')
    parser.add_argument('--output_dir', type=str, default='./data')
    args = parser.parse_args()

    # vggt 模型加载
    vggt_model = load_vggt_model()
    vggt_model.eval()

    input_root = os.path.join(args.pkl_root, f'{args.task_name}_{args.camera_type}_pkl')
    save_dir = os.path.join(args.output_dir, f'{args.task_name}_{args.camera_type}_{args.num_episodes}.zarr')

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    head_camera_arrays = []
    front_camera_arrays = []
    original_head_camera_arrays = []
    original_front_camera_arrays = []
    vggt_features_arrays = []
    vggt_features_current_arrays = []
    action_arrays = []
    state_arrays = []
    joint_action_arrays = []
    episode_ends_arrays = []

    total_count = 0
    current_ep = 0
    episode_list = list(range(args.num_episodes))
    print(episode_list)

    # tqdm 包 episode 层
    for current_ep in tqdm(episode_list, desc="Processing episodes"):
        epi_dir = f'{input_root}/episode{current_ep}'
        frame_files = sorted(
            glob.glob(f'{epi_dir}/*.pkl'),
            key=lambda x: int(os.path.basename(x).replace('.pkl',''))
        )

        frame_iter = tqdm(
            range(len(frame_files) - 1),
            desc=f"Episode {current_ep}",
            leave=False
        )

        for i in frame_iter:
            # t0
            with open(frame_files[i], 'rb') as f:
                data_t0 = pickle.load(f)
            # t1
            with open(frame_files[i+1], 'rb') as f:
                data_t1 = pickle.load(f)

            head_img_t0 = preprocess_rgb(data_t0['observation']['head_camera']['rgb'])
            front_img_t0 = preprocess_rgb(data_t0['observation']['front_camera']['rgb'])

            head_img_t1 = data_t1['observation']['head_camera']['rgb']
            front_img_t1 = data_t1['observation']['front_camera']['rgb']
            # head_img_t1 = np.transpose(head_img_t1.astype(np.float32)/255.0, (2, 0, 1))
            # front_img_t1 = np.transpose(front_img_t1.astype(np.float32)/255.0, (2, 0, 1))

            # VGGT 特征提取
            img_tensor_t0 = torch.tensor([head_img_t0,front_img_t0]).unsqueeze(0).cuda()   

            # test_vggt(vggt_model, img_tensor_t0)  # 测试 VGGT 模型

            vggt_feat_t0 = extract_features(vggt_model, img_tensor_t0)[0]

            # 存 t1 数据
            head_camera_arrays.append(head_img_t1)
            front_camera_arrays.append(front_img_t1)
            vggt_features_arrays.append(vggt_feat_t0)
            
            head_img_t1 = preprocess_rgb(head_img_t1)
            front_img_t1 = preprocess_rgb(front_img_t1)
            print("original_head_camera.shape: ", head_img_t1.shape)
            original_head_camera_arrays.append(head_img_t1)
            original_front_camera_arrays.append(front_img_t1)

            img_tensor_t1 = torch.tensor([head_img_t1, front_img_t1]).unsqueeze(0).cuda()
            vggt_feat_t1 = extract_features(vggt_model, img_tensor_t1)[0]
            vggt_features_current_arrays.append(vggt_feat_t1)

            action_arrays.append(data_t1['endpose'])
            state_arrays.append(data_t1['joint_action'])
            joint_action_arrays.append(data_t1['joint_action'])

            total_count += 1

        current_ep += 1
        episode_ends_arrays.append(total_count)

    # numpy 转换
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    front_camera_arrays = np.array(front_camera_arrays)
    original_head_camera_arrays = np.array(original_head_camera_arrays)
    original_front_camera_arrays = np.array(original_front_camera_arrays)
    vggt_features_arrays = np.array(vggt_features_arrays)
    vggt_features_current_arrays = np.array(vggt_features_current_arrays)

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    front_camera_arrays = np.moveaxis(front_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    chunk_size = 100

    zarr_data.create_dataset('head_camera', data=head_camera_arrays, chunks=(chunk_size, *head_camera_arrays.shape[1:]), overwrite=True, compressor=compressor)
    zarr_data.create_dataset('front_camera', data=front_camera_arrays, chunks=(chunk_size, *front_camera_arrays.shape[1:]), overwrite=True, compressor=compressor)
    zarr_data.create_dataset('vggt_head_camera', data=original_head_camera_arrays, chunks=(chunk_size, *original_head_camera_arrays.shape[1:]), overwrite=True, compressor=compressor)
    zarr_data.create_dataset('vggt_front_camera', data=original_front_camera_arrays, chunks=(chunk_size, *original_front_camera_arrays.shape[1:]), overwrite=True, compressor=compressor)
    zarr_data.create_dataset('vggt_features', data=vggt_features_arrays, chunks=(chunk_size, *vggt_features_arrays.shape[1:]), overwrite=True, compressor=compressor)
    zarr_data.create_dataset('vggt_features_current', data=vggt_features_current_arrays, chunks=(chunk_size, *vggt_features_current_arrays.shape[1:]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=(chunk_size, action_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=(chunk_size, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=(chunk_size, joint_action_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

    print(f"\n✅ Done! Zarr saved to {save_dir}")

if __name__ == '__main__':
    main()
