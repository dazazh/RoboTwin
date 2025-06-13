#!/bin/bash

echo "🎨 MultiImageObsEncoder Forward 可视化工具"
echo "============================================"

# 设置环境
export CUDA_VISIBLE_DEVICES=0

echo "📁 当前工作目录: $(pwd)"
echo "🐍 Python版本: $(python --version)"

# 检查是否在正确的目录
if [[ ! -f "quick_visualize.py" ]]; then
    echo "❌ 错误：请在 dp_multi 目录下运行此脚本"
    exit 1
fi

echo ""
echo "🚀 开始可视化..."

# 方法1: 快速测试（基础版本）
echo "📊 运行快速测试..."
python quick_visualize.py
if [ $? -eq 0 ]; then
    echo "✅ 快速测试完成"
else
    echo "❌ 快速测试失败"
fi

echo ""

# 方法2: 运行基础版本
echo "📊 运行基础可视化..."
python visualize_forward.py
if [ $? -eq 0 ]; then
    echo "✅ 基础可视化完成"
else
    echo "❌ 基础可视化失败"
fi

echo ""
echo "✅ 可视化完成！"
echo "📊 结果保存在 visualizations/ 目录下"
echo ""
echo "生成的文件："
ls -la visualizations/ 2>/dev/null || echo "❌ visualizations/ 目录不存在" 