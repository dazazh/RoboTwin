import pickle
from feature_pca import FeaturePCA
import numpy as np
import os
from dino import *
from tqdm import tqdm  # 导入 tqdm

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
camera = 'head_camera'
device = 'cuda'
episode_num = 30
n_components = 128
PCA_folder = "/data3/yuxuan/robotics/RoboTwin-main/policy/Diffusion-Policy-Dino-PCA/PCA_model"
PCA_model_path = PCA_folder + f'/pca_{n_components}_{episode_num}.pkl'
model = DINO(dino_name="dinov2", model_name='vits14').to(device)
total_feature = None

# 外层循环：遍历 episode
for episode_idx in tqdm(range(episode_num), desc="Processing episodes", unit="episode"):
    folder_path = f"/data3/yuxuan/robotics/RoboTwin-main/data/tube_grasp_D435_pkl/episode{episode_idx}"
    counter = 0
    # 内层循环：遍历文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".pkl") and counter % 1 == 0:  # 每10个文件采样一次
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            color = data['observation'][camera]['rgb'][..., :3]
            image = np.array(color)
            H, W, _ = image.shape
            feature_map, feature_line = get_dino_feature(image, transform_size=420, model=model)
            # 应用分块池化以降低特征维度
            # 假设 feature_map 的形状为 [30, 30,384] 或其他类似形状，需要根据实际情况调整
            pooled_feature = block_pooling(feature_map, block_size=5, block_size_C=8)  # 调整 block_size 以控制降维幅度
            # 将池化后的特征展平为一维
            feature_line = pooled_feature.reshape(1, -1)
            if total_feature is None:
                total_feature = feature_line
            else:
                total_feature = torch.cat((total_feature, feature_line), dim=0)
        counter += 1


# 拟合 PCA 模型
total_pca = FeaturePCA(n_components=n_components)
total_pca.fit(total_feature.cpu())
total_pca.save_pca(PCA_model_path)