import pickle
import os
import sys
import numpy as np

def read_pkl_file(file_path):
    """
    读取pkl文件并返回其内容
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def print_array_info(arr, name=""):
    """打印数组的详细信息"""
    if isinstance(arr, np.ndarray):
        print(f"\n{name}信息:")
        print(f"形状: {arr.shape}")
        print(f"数据类型: {arr.dtype}")
        print(f"数据类型精度: {arr.dtype.itemsize * 8} bits")
        print(f"数据类型范围: {np.finfo(arr.dtype) if np.issubdtype(arr.dtype, np.floating) else np.iinfo(arr.dtype)}")
    elif isinstance(arr, list):
        print(f"\n{name}信息 (列表):")
        print(f"列表长度: {len(arr)}")
        for i, item in enumerate(arr):
            if isinstance(item, np.ndarray):
                print(f"\n第{i}个元素:")
                print(f"形状: {item.shape}")
                print(f"数据类型: {item.dtype}")
                print(f"数据类型精度: {item.dtype.itemsize * 8} bits")
                print(f"数据类型范围: {np.finfo(item.dtype) if np.issubdtype(item.dtype, np.floating) else np.iinfo(item.dtype)}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_pkl.py <path_to_pkl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    data = read_pkl_file(file_path)
    if data is None:
        sys.exit(1)
    
    # 打印aggregated_tokens_list的信息
    if 'aggregated_tokens_list' in data:
        print_array_info(data['aggregated_tokens_list'], "aggregated_tokens_list")
    else:
        print("No 'aggregated_tokens_list' found in the data")
    
    # 打印patch_start_idx的信息
    if 'patch_start_idx' in data:
        print_array_info(data['patch_start_idx'], "patch_start_idx")
    else:
        print("No 'patch_start_idx' found in the data")

if __name__ == "__main__":
    main() 