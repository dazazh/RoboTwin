import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    
    # # 只冻结卷积层
    # for name, param in resnet.named_parameters():
    #     if "fc" not in name:  # 不冻结全连接层
    #         param.requires_grad = False
    
    # # 打印参数状态
    # print("Parameter status after freezing:")
    # for name, param in resnet.named_parameters():
    #     print(f"{name}: requires_grad = {param.requires_grad}")
    
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def visualize_feature_maps(model, input_image):
    # 确保模型在评估模式
    model.eval()
    
    # 注册hook来获取特征图
    feature_maps = {}
    def get_features(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()
        return hook
    
    # 为每个感兴趣的层注册hook
    model.layer1.register_forward_hook(get_features('layer1'))
    model.layer2.register_forward_hook(get_features('layer2'))
    model.layer3.register_forward_hook(get_features('layer3'))
    model.layer4.register_forward_hook(get_features('layer4'))
    
    # 前向传播
    with torch.no_grad():
        _ = model(input_image)
    
    # 可视化每个层的特征图
    for layer_name, features in feature_maps.items():
        # 获取第一个batch的特征图
        print("features.shape: ",features.shape)
        feature_map = features[0]
        
        # 计算要显示的通道数（最多显示16个通道）
        num_channels = min(16, feature_map.shape[0])
        
        # 创建子图
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f'{layer_name} Feature Maps')
        
        # 显示每个通道的特征图
        for i in range(num_channels):
            row = i // 4
            col = i % 4
            channel_data = feature_map[i].cpu().numpy()
            
            # 归一化到[0,1]范围
            channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
            
            axes[row, col].imshow(channel_data, cmap='viridis')
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Channel {i}')
        
        plt.tight_layout()
        plt.savefig(f'{layer_name}_feature_maps.png')
        plt.close()

def check_model_frozen(model):
    """
    检查模型参数是否被冻结
    """
    print("\nChecking model parameter status:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")
    
    # 计算冻结和未冻结的参数数量
    frozen_params = sum(1 for param in model.parameters() if not param.requires_grad)
    total_params = sum(1 for param in model.parameters())
    print(f"\nFrozen parameters: {frozen_params}/{total_params} ({frozen_params/total_params*100:.2f}%)")

if __name__ == "__main__":
    model = get_resnet("resnet18", weights="IMAGENET1K_V1")
    print(model)
    input_image = cv2.imread("/mnt/workspace/yuhao/depth_encoder_test/vggt/examples/llff_fern/images/000.png")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    visualize_feature_maps(model, input_image)
