import torch
import numpy as np
import matplotlib.pyplot as plt

def test_vggt(model, images):
    print('imges shape:', images.shape)
    B, S, C, H, W = images.shape
    N = 100  # number of query points
    # Create a grid of points
    x = torch.linspace(0, W-1, int(np.sqrt(N)), dtype=torch.float32)
    y = torch.linspace(0, H-1, int(np.sqrt(N)), dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    query_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:N]  # [N, 2]

    # Expand to match batch dimension
    query_points = query_points.unsqueeze(0).expand(B, N, 2).to(device)  # [B, N, 2]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images, query_points=query_points)

    # Convert predictions to float32 before converting to numpy
    depth_maps = predictions["depth"].to(torch.float32).cpu().numpy()  # Shape: [B, S, H, W, 1]
    depth_confs = predictions["depth_conf"].to(torch.float32).cpu().numpy()  # Shape: [B, S, H, W]
    tracks = predictions["track"].to(torch.float32).cpu().numpy()  # Shape: [B, S, N, 2]
    vis = predictions["vis"].to(torch.float32).cpu().numpy()  # Shape: [B, S, N]
    conf = predictions["conf"].to(torch.float32).cpu().numpy()  # Shape: [B, S, N]

    # Create a figure with subplots for each image and its depth map
    n_images = depth_maps.shape[1]  # number of images in sequence
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 5*n_images))

    for i in range(n_images):
        # Get depth map and confidence for this image
        depth = depth_maps[0, i, :, :, 0]  # Remove batch and channel dimensions
        conf_map = depth_confs[0, i]
        
        # Get tracked points for this image
        points = tracks[0, i]  # [N, 2]
        visibility = vis[0, i]  # [N]
        point_conf = conf[0, i]  # [N]
        
        # Plot depth map with tracked points
        im = axes[i, 0].imshow(depth, cmap='viridis')
        # Plot points with visibility > 0.5
        valid_points = points[visibility > 0.5]
        if len(valid_points) > 0:
            axes[i, 0].scatter(valid_points[:, 0], valid_points[:, 1], c='red', s=10, alpha=0.5)
        axes[i, 0].set_title(f'Depth Map {i} with Tracked Points')
        plt.colorbar(im, ax=axes[i, 0])
        
        # Plot confidence map
        im = axes[i, 1].imshow(conf_map, cmap='hot')
        axes[i, 1].set_title(f'Confidence Map {i}')
        plt.colorbar(im, ax=axes[i, 1])

    plt.tight_layout()
    plt.savefig('depth_visualization_with_tracks.png')
    plt.close()
    assert False, "Test VGGT model completed, check the saved image for depth and tracked points visualization."
            