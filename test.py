from segment_anything import sam_model_registry
from PIL import Image
import numpy as np
import torch
import time

def sam_encoder(model_type, checkpoint_path, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    image_encoder = sam.image_encoder
    image_encoder.to(device)
    return image_encoder

def image_preprocess(image_path, device):
  # 输入图片路径
    image = Image.open(image_path).convert("RGB")
    # 调整图像大小到1024x1024
    image = image.resize((1024, 1024), Image.Resampling.BILINEAR)
    image = np.array(image)  # 转换为NumPy数组
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)  # 转换为张量并调整维度[B, C, H, W]
    # 归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std  # 归一化
    image_tensor = image_tensor.to(device)
    print(image_tensor.shape)
    return image_tensor

def sam_features(model_type, image_path, checkpoint_path, device):
    image_encoder = sam_encoder(model_type, checkpoint_path, device) 
    image_tensor = image_preprocess(image_path, device)
    with torch.no_grad():  # 禁用梯度计算
        features = image_encoder(image_tensor)
        print("Original features shape:", features.shape)
        features = features.flatten(1)
        return features
    
if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_type = 'vit_b'
    checkpoint_path = "./segment-anything/checkpoint/sam_vit_b_01ec64.pth"  # 预训练权重文件
    image_path = "./test.JPG"
    start_time = time.time()  # 开始时间
    features = sam_features(model_type, image_path, checkpoint_path, device)
    end_time = time.time()  # 结束时间
    # 打印推理时间
    inference_time = end_time - start_time
    print("Device:", device)
    print("Inference time:{:.4f} seconds".format(inference_time))
    print("Extracted features shape:", features.shape)