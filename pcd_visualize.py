import open3d as o3d
import os

# 设置包含PCD文件的目录路径
pcd_directory = "/home/robot4chem/robot/RoboTwin/data/tube_grasp_D435/episode0/camera/pointCloud/head"

# 获取所有.pcd文件并按数字排序
pcd_files = sorted([f for f in os.listdir(pcd_directory) if f.endswith('.pcd')],
                   key=lambda x: int(x.split('.')[0]))  # 假设文件名格式为 0.pcd, 1.pcd, ...

# 循环遍历文件并逐个显示
for id,pcd_file in enumerate(pcd_files):
    pcd_path = os.path.join(pcd_directory, pcd_file)
    print(f"Loading {pcd_file}...")
    if id%10==0:
        # 加载PCD文件
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        # 可视化点云
        o3d.visualization.draw_geometries([pcd])
