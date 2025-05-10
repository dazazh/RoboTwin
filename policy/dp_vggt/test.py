import torch
import pickle
import os
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
import matplotlib.pyplot as plt
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT().to(device)
ckpt = torch.load('/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/policy/dp_vggt/vggt/checkpoints/original_model.pt')
model.load_state_dict(ckpt,strict=False)
# Load and preprocess example images (replace with your own image paths)
image_names1 = ["/mnt/workspace/yuhao/vggt/examples/llff_fern/images/000.png", "/mnt/workspace/yuhao/vggt/examples/llff_fern/images/001.png"]  
image_names2 = ["/mnt/workspace/yuhao/vggt/examples/kitchen/images/02.png", "/mnt/workspace/yuhao/vggt/examples/kitchen/images/03.png"]  

# 创建32组图像对
batch_size = 32
images_list = []

for i in range(batch_size):
    # 加载第一组图像
    images1 = load_and_preprocess_images(image_names1).to(device)
    images1 = images1.squeeze(0)
    images_list.append(images1)

# 将所有图像堆叠成一个大batch
images = torch.stack(images_list, dim=0)
print("images.shape: ", images.shape)  # 应该是 [32, 4, 3, H, W]
# print(model)

# from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from vggt.utils.geometry import unproject_depth_map_to_point_map

# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         aggregated_tokens_list, ps_idx = model.aggregator(images)
                
#     # Predict Cameras
#     pose_enc = model.camera_head(aggregated_tokens_list)[-1]
#     # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
#     extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

#     # Predict Depth Maps
#     depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

#     # Predict Point Maps
#     point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
#     # Construct 3D Points from Depth Maps and Cameras
#     # which usually leads to more accurate 3D points than point map branch
#     point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
#                                                                 extrinsic.squeeze(0), 
#                                                                 intrinsic.squeeze(0))

#     # Predict Tracks
#     # choose your own points to track, with shape (N, 2) for one scene
#     query_points = torch.FloatTensor([[100.0, 200.0], 
#                                         [60.72, 259.94]]).to(device)
#     track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
#     depths = predictions['depth']
    
#     # depths = depth.squeeze(0)
#     print("depth.shape: ",depths.shape)
    
#     depths = depths[1]
#     print("depth.shape: ",depths.shape)
#     depths = np.array(depths.cpu())

# depths = np.squeeze(depths, axis=-1)  # 去掉最后一个维度 (1)

# 遍历三张深度图并生成热图并保存
# for i in range(2):
#     depth_map = depths[i]  # 获取当前深度图

#     # 将深度图转换为一个合适的范围 (0到255)
#     depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#     depth_map_normalized = np.uint8(depth_map_normalized)

#     # 使用 OpenCV 的 applyColorMap 将深度图转换为热图
#     heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)  # 你可以选择不同的色图

#     # 保存热图为文件
#     filename = f"depth{i}.png"
#     cv2.imwrite(filename, heatmap)

#     print(f"Saved heatmap as {filename}")


import matplotlib.pyplot as plt
import numpy as np
import torch

# 假设特征图是PyTorch张量，形状为[batchsize, 3, 925, 2048]
# 示例数据（替换为实际特征图）
feature_map = predictions
print("feature_map.shape: ",feature_map.shape)

# 提取第一个batch并转换为numpy数组
first_batch = feature_map[1].detach().cpu().numpy()  # 形状 (3, 925, 2048)

# 遍历三个通道
for channel in range(3):
    # 获取当前通道数据
    channel_data = first_batch[channel]
    
    # 归一化到 [0, 1]
    min_val = np.min(channel_data)
    max_val = np.max(channel_data)
    normalized = (channel_data - min_val) / (max_val - min_val + 1e-8)  # 避免除以零
    
    # 创建图像
    # plt.figure(figsize=(20, 5))  # 调整图像宽度以适应长宽比
    plt.imshow(normalized, cmap='hot', aspect='auto')  # 使用热力图颜色映射
    plt.axis('off')  # 关闭坐标轴
    
    # 保存为PNG文件
    plt.savefig(f'channel_{channel}_heatmap.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()  # 关闭图形释放内存

def process_pkl_with_vggt(pkl_path, output_path, model):
    # 加载pkl文件
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 获取图像数据
    head_cam = data['observation']['head_camera']['rgb']
    front_cam = data['observation']['front_camera']['rgb']
    
    # 转换为tensor并预处理
    head_cam = torch.from_numpy(head_cam).float() / 255.0
    front_cam = torch.from_numpy(front_cam).float() / 255.0
    
    # 确保在正确的设备上
    device = next(model.parameters()).device
    head_cam = head_cam.to(device)
    front_cam = front_cam.to(device)
    
    # 堆叠图像
    images = torch.stack([head_cam, front_cam], dim=1)
    
    # 获取VGGT的中间层特征
    with torch.no_grad():
        # 获取aggregator的输出
        aggregated_tokens_list, patch_start_idx = model.aggregator(images)
        
        # 获取adapter_head的输出
        vggt_features = model.adapter_head(
            aggregated_tokens_list, 
            images=images, 
            patch_start_idx=patch_start_idx
        )
    
    # 将特征转换为numpy
    vggt_features = vggt_features.cpu().numpy()
    
    # 更新数据字典
    data['vggt_features'] = vggt_features
    data['head_camera_rgb'] = head_cam.cpu().numpy()
    data['front_camera_rgb'] = front_cam.cpu().numpy()
    
    # 保存更新后的数据
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def main():
    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT().to(device)
    ckpt = torch.load('/mnt/workspace/yuhao/depth_encoder_test/RoboTwin-encoder/policy/dp_vggt/vggt/checkpoints/original_model.pt')
    model.load_state_dict(ckpt, strict=False)
    
    # 设置输入输出路径
    input_dir = "path/to/input/pkl/files"
    output_dir = "path/to/output/pkl/files"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有pkl文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.pkl'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_pkl_with_vggt(input_path, output_path, model)
            print(f"Processed {filename}")

if __name__ == "__main__":
    main()