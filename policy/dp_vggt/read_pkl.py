import pickle
import os
import sys

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
    
    if 'vggt_features' in data:
        features = data['vggt_features']
        print(f"Features shape: {features.shape if hasattr(features, 'shape') else 'No shape attribute'}")
        print(f"Features type: {type(features)}")
    else:
        print("No 'features' key found in the data")

if __name__ == "__main__":
    main()