 #!/usr/bin/env python3
"""
调试工具：检查路径和数据文件
"""

import os
import sys
from pathlib import Path

def check_environment():
    """检查环境和路径"""
    print("🔍 环境检查")
    print("="*50)
    
    # 当前工作目录
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"🐍 Python路径: {sys.executable}")
    
    # 检查关键文件
    key_files = [
        "quick_visualize.py",
        "visualize_forward.py", 
        "diffusion_policy/",
        "checkpoints/",
        "data/"
    ]
    
    print(f"\n📋 关键文件检查:")
    for file in key_files:
        exists = Path(file).exists()
        print(f"   {'✅' if exists else '❌'} {file}")
    
    # 检查checkpoint
    print(f"\n🎯 Checkpoint检查:")
    if Path("checkpoints").exists():
        for root, dirs, files in os.walk("checkpoints"):
            for file in files:
                if file.endswith('.ckpt'):
                    full_path = os.path.join(root, file)
                    size = os.path.getsize(full_path) / (1024*1024)  # MB
                    print(f"   ✅ {full_path} ({size:.1f} MB)")
    else:
        print("   ❌ checkpoints/ 目录不存在")
    
    # 检查数据文件
    print(f"\n💾 数据文件检查:")
    data_paths = [
        "data/",
        "../data/",
        "../../data/",
        "../../../data/"
    ]
    
    for data_path in data_paths:
        if Path(data_path).exists():
            print(f"   ✅ {data_path} 存在")
            for item in Path(data_path).iterdir():
                if item.name.endswith('.zarr'):
                    print(f"      📊 {item.name}")
        else:
            print(f"   ❌ {data_path} 不存在")

def test_imports():
    """测试导入"""
    print(f"\n🔬 导入测试:")
    
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"   ❌ PyTorch导入失败: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print(f"   ✅ Matplotlib 可用")
    except Exception as e:
        print(f"   ❌ Matplotlib导入失败: {e}")
    
    try:
        sys.path.append('../..')
        sys.path.insert(0, '../Multi-Diffusion-Policy')
        from diffusion_policy.workspace.robotworkspace import RobotWorkspace
        print(f"   ✅ Diffusion Policy 可用")
    except Exception as e:
        print(f"   ❌ Diffusion Policy导入失败: {e}")

def main():
    print("🛠️  Dp_multi 调试工具")
    print("="*60)
    
    check_environment()
    test_imports()
    
    print(f"\n🎯 建议:")
    print("  1. 确保在 dp_multi 目录下运行脚本")
    print("  2. 确保 zarr 数据文件存在")
    print("  3. 确保 checkpoint 文件存在")
    print("  4. 运行: python test_debug.py")
    print("  5. 然后运行: python quick_visualize.py")

if __name__ == "__main__":
    main()