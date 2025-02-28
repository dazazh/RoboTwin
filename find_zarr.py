import zarr
import numpy as np

# 打开 Zarr 文件
zarr_file = zarr.open('/mnt/workspace/yuhao/RoboTwin/policy/Depth-Distillation/data/tube_grasp_D435_300.zarr', mode='r')  # 替换为你的 Zarr 文件路径

# 假设你要访问的数据在数组的某个位置
# 比如，读取索引位置 [100, 200, 300] 的元素
data = zarr_file['data']['head_depth']
count_zeros = np.max(data)
print(count_zeros)