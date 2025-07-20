#!/bin/bash

# 启动Mininet多线程MLT协议实验脚本

echo "=== 启动Mininet多线程MLT协议实验 ==="

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then
    echo "错误: 请以root权限运行此脚本 (使用 sudo)"
    exit 1
fi

# 检查Mininet是否安装
if ! command -v mn &> /dev/null; then
    echo "错误: Mininet未安装。请先安装Mininet"
    exit 1
fi

# 检查Python依赖
echo "检查Python依赖..."
python3 -c "import torch, torchvision, transformers, sklearn, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python依赖包"
    echo "请安装: torch, torchvision, transformers, sklearn, numpy"
    exit 1
fi

# 检查数据文件是否存在
if [ ! -d "data/cifar10_splits" ]; then
    echo "错误: 数据目录不存在。请先运行 prepare_data.py 生成CIFAR10数据分割"
    exit 1
fi

# 创建日志目录
mkdir -p logs

echo "启动Mininet拓扑..."
echo "注意: 实验将在Mininet环境中运行"
echo "使用 'exit' 命令退出Mininet CLI并停止实验"
echo ""

# 运行Mininet拓扑
python3 mininet/multithreading_topo.py

echo ""
echo "=== 实验完成 ==="
echo "日志文件保存在 logs/ 目录中:"
echo "  - server.log: 服务器日志"
echo "  - worker0.log: 工作节点0日志"
echo "  - worker1.log: 工作节点1日志"
echo "  - worker2.log: 工作节点2日志" 