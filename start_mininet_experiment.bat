@echo off
REM 启动Mininet多线程MLT协议实验脚本 (Windows版本)

echo === 启动Mininet多线程MLT协议实验 ===

REM 检查是否以管理员权限运行
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 错误: 请以管理员权限运行此脚本
    echo 右键点击此脚本，选择"以管理员身份运行"
    pause
    exit /b 1
)

REM 检查Mininet是否安装
where mn >nul 2>&1
if %errorLevel% neq 0 (
    echo 错误: Mininet未安装。请先安装Mininet
    pause
    exit /b 1
)

REM 检查Python依赖
echo 检查Python依赖...
python -c "import torch, torchvision, transformers, sklearn, numpy" 2>nul
if %errorLevel% neq 0 (
    echo 错误: 缺少必要的Python依赖包
    echo 请安装: torch, torchvision, transformers, sklearn, numpy
    pause
    exit /b 1
)

REM 检查数据文件是否存在
if not exist "data\cifar10_splits" (
    echo 错误: 数据目录不存在。请先运行 prepare_data.py 生成CIFAR10数据分割
    pause
    exit /b 1
)

REM 创建日志目录
if not exist "logs" mkdir logs

echo 启动Mininet拓扑...
echo 注意: 实验将在Mininet环境中运行
echo 使用 'exit' 命令退出Mininet CLI并停止实验
echo.

REM 运行Mininet拓扑
python mininet\multithreading_topo.py

echo.
echo === 实验完成 ===
echo 日志文件保存在 logs\ 目录中:
echo   - server.log: 服务器日志
echo   - worker0.log: 工作节点0日志
echo   - worker1.log: 工作节点1日志
echo   - worker2.log: 工作节点2日志

pause 