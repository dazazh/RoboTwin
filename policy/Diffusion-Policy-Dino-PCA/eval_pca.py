import pickle
from feature_pca import FeaturePCA
import numpy as np
import os
from dino import *
from tqdm import tqdm  # 导入 tqdm
import time

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

PCA_model_path = PCA_model_path 
feature_pca = FeaturePCA(n_components)
feature_pca.load_pca(PCA_model_path)

start_time = time.time()
# folder_path = f"/data3/yuxuan/robotics/RoboTwin-main/data/tube_grasp_D435_pkl/episode0"
# file_path = os.path.join(folder_path, '0.pkl')
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
# color = data['observation'][camera]['rgb'][..., :3]
# image = np.array(color)
# H, W, _ = image.shape
# feature_map, feature_line = get_dino_feature(image, transform_size=420, model=model)
# pooled_feature = block_pooling(feature_map, block_size=40, block_size_C=16)  # 调整 block_size 以控制降维幅度
# # 将池化后的特征展平为一维
# feature_line = pooled_feature.reshape(1, -1)
# feature_line = feature_pca.transform(feature_line)
# end_time = time.time()
# print("inference time: ", end_time-start_time)
# print(feature_line.shape)

# image_batch = torch.randn(384, 3, 240, 320)
# sam_features = get_dino_feature_batch(image_batch, 420, model)
# pooled_features = block_pooling_batch(sam_features, block_size=5, block_size_C=8) 
# feature_line = feature_pca.transform(pooled_features)
# print(feature_line.shape)

print(model)