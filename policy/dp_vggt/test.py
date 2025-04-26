import torch
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
model.load_state_dict(ckpt,strict=True)
# Load and preprocess example images (replace with your own image paths)
image_names = ["/mnt/workspace/yuhao/vggt/examples/kitchen/images/00.png", "/mnt/workspace/yuhao/vggt/examples/kitchen/images/01.png", "/mnt/workspace/yuhao/vggt/examples/kitchen/images/02.png"]  
images = load_and_preprocess_images(image_names).to(device)
images = images.to(device)
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
    depth = predictions['depth']
    depths = depth.squeeze(0)
    depths = np.array(depths.cpu())

depths = np.squeeze(depths, axis=-1)  # 去掉最后一个维度 (1)

# 遍历三张深度图并生成热图并保存
for i in range(3):
    depth_map = depths[i]  # 获取当前深度图

    # 将深度图转换为一个合适的范围 (0到255)
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = np.uint8(depth_map_normalized)

    # 使用 OpenCV 的 applyColorMap 将深度图转换为热图
    heatmap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)  # 你可以选择不同的色图

    # 保存热图为文件
    filename = f"depth{i}.png"
    cv2.imwrite(filename, heatmap)

    print(f"Saved heatmap as {filename}")