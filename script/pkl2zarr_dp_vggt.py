import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import einops
import cv2


def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='block_hammer_beat',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('expert_data_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    head_camera_type = args.head_camera_type
    load_dir = f'/cpfs04/shared/muyao/yuhao/data/{task_name}_{head_camera_type}_with_embedding_pkl'
    
    total_count = 0

    save_dir = f'/cpfs04/shared/muyao/yuhao/data_zarr/{task_name}_{head_camera_type}_300.zarr'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    # First pass to count total number of frames
    total_frames = 0
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        file_num = 0
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            total_frames += 1
            file_num += 1
        current_ep += 1
    
    current_ep = 0
    total_count = 0
    episode_ends = []

    # Initialize zarr datasets with known shapes
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # Get shape of first frame to initialize datasets
    with open(load_dir+f'/episode0/0.pkl', 'rb') as file:
        first_data = pickle.load(file)
        head_img_shape = first_data['observation']['head_camera']['rgb'].shape
        front_img_shape = first_data['observation']['front_camera']['rgb'].shape
        vggt_features_shape = first_data['vggt_features'].shape
        action_shape = first_data['endpose'].shape
        joint_action_shape = first_data['joint_action'].shape

    # Create datasets with appropriate shapes
    zarr_head_camera = zarr_data.create_dataset('head_camera', 
                                          shape=(total_frames, *head_img_shape),
                                          chunks=(100, *head_img_shape),
                                          dtype='uint8',
                                          compressor=compressor)
    
    zarr_front_camera = zarr_data.create_dataset('front_camera',
                                           shape=(total_frames, *front_img_shape),
                                           chunks=(100, *front_img_shape),
                                           dtype='uint8',
                                           compressor=compressor)
    
    zarr_vggt_features = zarr_data.create_dataset('vggt_features',
                                            shape=(total_frames, *vggt_features_shape),
                                            chunks=(100, *vggt_features_shape),
                                            dtype='float32',
                                            compressor=compressor)
    
    zarr_tcp_action = zarr_data.create_dataset('tcp_action',
                                         shape=(total_frames, *action_shape),
                                         chunks=(100, *action_shape),
                                         dtype='float32',
                                         compressor=compressor)
    
    zarr_state = zarr_data.create_dataset('state',
                                    shape=(total_frames, *joint_action_shape),
                                    chunks=(100, *joint_action_shape),
                                    dtype='float32',
                                    compressor=compressor)
    
    zarr_action = zarr_data.create_dataset('action',
                                     shape=(total_frames, *joint_action_shape),
                                     chunks=(100, *joint_action_shape),
                                     dtype='float32',
                                     compressor=compressor)

    # Process episodes and write data in batches
    BATCH_SIZE = 100  # Match with chunk size
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        
        # Initialize batch arrays
        batch_head_imgs = []
        batch_front_imgs = []
        batch_vggt_features = []
        batch_actions = []
        batch_joint_actions = []
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            batch_head_imgs.append(data['observation']['head_camera']['rgb'])
            batch_front_imgs.append(data['observation']['front_camera']['rgb'])
            batch_vggt_features.append(data['vggt_features'])
            batch_actions.append(data['endpose'])
            batch_joint_actions.append(data['joint_action'])

            del data
            file_num += 1
            total_count += 1
            
            # Write batch when it's full or at the end of episode
            if len(batch_head_imgs) == BATCH_SIZE or not os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
                start_idx = total_count - len(batch_head_imgs)
                
                # Convert lists to numpy arrays and reshape
                batch_head_imgs = np.stack(batch_head_imgs)
                batch_front_imgs = np.stack(batch_front_imgs)
                batch_vggt_features = np.stack(batch_vggt_features)
                batch_actions = np.stack(batch_actions)
                batch_joint_actions = np.stack(batch_joint_actions)
                
                # Write batch to zarr datasets
                zarr_head_camera[start_idx:start_idx + len(batch_head_imgs)] = batch_head_imgs
                zarr_front_camera[start_idx:start_idx + len(batch_front_imgs)] = batch_front_imgs
                zarr_vggt_features[start_idx:start_idx + len(batch_vggt_features)] = batch_vggt_features
                zarr_tcp_action[start_idx:start_idx + len(batch_actions)] = batch_actions
                zarr_state[start_idx:start_idx + len(batch_joint_actions)] = batch_joint_actions
                zarr_action[start_idx:start_idx + len(batch_joint_actions)] = batch_joint_actions
                
                # Clear batch arrays
                batch_head_imgs = []
                batch_front_imgs = []
                batch_vggt_features = []
                batch_actions = []
                batch_joint_actions = []
        
        episode_ends.append(total_count)
        current_ep += 1

    print()
    # Save episode ends
    zarr_meta.create_dataset('episode_ends', 
                           data=np.array(episode_ends),
                           dtype='int64',
                           compressor=compressor)

if __name__ == '__main__':
    main()
