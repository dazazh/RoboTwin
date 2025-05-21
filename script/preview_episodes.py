import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    # 输入输出路径
    input_dir = "/data/user/xcs/yuhao/3d-aware/RoboTwin/data/transparent_cup_place_L515_pkl"
    output_dir = "./data/preview"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有episode文件夹
    episode_dirs = [d for d in os.listdir(input_dir) if d.startswith('episode')]
    episode_dirs.sort()  # 按episode编号排序

    # 处理每个episode
    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        # 获取episode编号
        episode_num = int(episode_dir.split('episode')[1])
        
        # 读取第一个pkl文件
        episode_path = os.path.join(input_dir, episode_dir)
        pkl_files = [f for f in os.listdir(episode_path) if f.endswith('.pkl')]
        pkl_files.sort()
        
        if not pkl_files:
            print(f"No pkl files found in {episode_dir}")
            continue
            
        first_pkl = os.path.join(episode_path, pkl_files[0])
        
        # 读取图片
        with open(first_pkl, 'rb') as f:
            data = pickle.load(f)
            head_cam = data['observation']['head_camera']['rgb']
            
        # 保存图片
        output_path = os.path.join(output_dir, f"episode{episode_num}.jpg")
        Image.fromarray(head_cam).save(output_path)

if __name__ == "__main__":
    main() 