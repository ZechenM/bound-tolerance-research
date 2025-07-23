#!/usr/bin/env python3
"""
在Mininet中测试MLT协议和ML训练 - 3个工作节点版本
"""

import os
import time
import subprocess
from mininet.net import Mininet
from mininet.log import info, setLogLevel

def test_mlt_training():
    """在Mininet中测试MLT训练 - 临时方案：1个Worker在独立host，2个在Server本地"""
    
    # 创建简单网络
    net = Mininet()
    
    # 添加主机：1个服务器 + 1个独立的工作节点
    server = net.addHost('server', ip='10.0.0.1/24')
    worker0 = net.addHost('worker0', ip='10.0.0.2/24')  # 独立的worker
    
    # 直接连接
    net.addLink(server, worker0)
    
    # 启动网络
    net.start()
    
    info("*** 网络拓扑: Server (含2个本地Workers) + 1个独立Worker host\n")
    
    project_dir = "/home/ubuntu/bound-tolerance-research"
    venv_python = f"{project_dir}/venv/bin/python3"
    
    info("*** 测试ML训练环境 - 3个工作节点\n")
    
    # 1. 测试数据是否存在
    info("1. 检查训练数据...\n")
    result = server.cmd(f"ls {project_dir}/data/cifar10_splits/")
    info(f"数据文件: {result}")
    
    # 2. 测试Python包导入
    info("2. 测试Python包导入...\n")
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import transformers; print(f'Transformers: {transformers.__version__}')",
        "import numpy as np; print(f'NumPy: {np.__version__}')"
    ]
    
    for test_cmd in test_imports:
        result = server.cmd(f"cd {project_dir} && {venv_python} -c \"{test_cmd}\"")
        info(f"导入测试: {result.strip()}\n")
    
    # 创建日志目录
    server.cmd(f"mkdir -p {project_dir}/logs")
    
    # 3. 启动服务器（后台）
    info("3. 启动MLT服务器...\n")
    server_cmd = f"cd {project_dir} && {venv_python} server_multithreading.py > {project_dir}/logs/server.log 2>&1 &"
    server.cmd(server_cmd)
    
    # 等待服务器启动
    time.sleep(5)
    
    # 检查服务器是否启动
    result = server.cmd("ps aux | grep server_multithreading | grep -v grep")
    if result.strip():
        info("✅ 服务器启动成功\n")
        info(f"服务器进程: {result.strip()}\n")
    else:
        info("❌ 服务器启动失败\n")
        # 显示错误日志
        error_log = server.cmd(f"cat {project_dir}/logs/server.log")
        info(f"服务器错误日志:\n{error_log}\n")
        net.stop()
        return
    
    # 4. 启动3个workers (1个在独立host，2个在server本地)
    info("4. 启动3个Workers (混合模式)...\n")
    worker_processes = []
    
    # Worker 0: 在独立的host上运行
    info("启动Worker 0 (独立host)...\n")
    worker0_cmd = f"cd {project_dir} && SERVER_HOST=10.0.0.1 {venv_python} worker_multithreading.py 0 > {project_dir}/logs/worker0.log 2>&1 &"
    worker0.cmd(worker0_cmd)
    
    time.sleep(3)
    result = worker0.cmd("ps aux | grep worker_multithreading | grep -v grep")
    if result.strip():
        info("✅ Worker0 (独立host) 启动成功\n")
        info(f"Worker0 进程: {result.strip()}\n")
        worker_processes.append(True)
    else:
        info("❌ Worker0 启动失败\n")
        error_log = worker0.cmd(f"cat {project_dir}/logs/worker0.log")
        info(f"Worker0 错误日志:\n{error_log}\n")
        worker_processes.append(False)
    
    # Worker 1 & 2: 在server本地运行
    info("启动Worker 1 (server本地)...\n")
    worker1_cmd = f"cd {project_dir} && SERVER_HOST=127.0.0.1 {venv_python} worker_multithreading.py 1 > {project_dir}/logs/worker1.log 2>&1 &"
    server.cmd(worker1_cmd)
    
    time.sleep(3)
    result = server.cmd("ps aux | grep 'worker_multithreading.py 1' | grep -v grep")
    if result.strip():
        info("✅ Worker1 (server本地) 启动成功\n")
        info(f"Worker1 进程: {result.strip()}\n")
        worker_processes.append(True)
    else:
        info("❌ Worker1 启动失败\n")
        error_log = server.cmd(f"cat {project_dir}/logs/worker1.log")
        info(f"Worker1 错误日志:\n{error_log}\n")
        worker_processes.append(False)
    
    info("启动Worker 2 (server本地)...\n")
    worker2_cmd = f"cd {project_dir} && SERVER_HOST=127.0.0.1 {venv_python} worker_multithreading.py 2 > {project_dir}/logs/worker2.log 2>&1 &"
    server.cmd(worker2_cmd)
    
    time.sleep(3)
    result = server.cmd("ps aux | grep 'worker_multithreading.py 2' | grep -v grep")
    if result.strip():
        info("✅ Worker2 (server本地) 启动成功\n")
        info(f"Worker2 进程: {result.strip()}\n")
        worker_processes.append(True)
    else:
        info("❌ Worker2 启动失败\n")
        error_log = server.cmd(f"cat {project_dir}/logs/worker2.log")
        info(f"Worker2 错误日志:\n{error_log}\n")
        worker_processes.append(False)
    
    successful_workers = sum(worker_processes)
    info(f"成功启动的Worker数量: {successful_workers}/3\n")
    info("配置说明: Worker0在独立host(10.0.0.2), Worker1&2在Server本地(127.0.0.1)\n")
    
    # 5. 监控训练过程直到完成
    info("5. 持续监控训练过程直到完成...\n")
    info("预计需要约30分钟，请耐心等待\n")
    
    check_count = 0
    while True:
        time.sleep(10)  # 每10秒检查一次
        check_count += 1
        
        # 检查进程状态
        server_alive = server.cmd("ps aux | grep server_multithreading | grep -v grep").strip()
        
        # 检查各个worker状态
        worker0_alive = worker0.cmd("ps aux | grep worker_multithreading | grep -v grep").strip()
        worker1_alive = server.cmd("ps aux | grep 'worker_multithreading.py 1' | grep -v grep").strip()
        worker2_alive = server.cmd("ps aux | grep 'worker_multithreading.py 2' | grep -v grep").strip()
        
        workers_alive = [bool(worker0_alive), bool(worker1_alive), bool(worker2_alive)]
        alive_count = sum(workers_alive)
        
        # 每分钟显示一次状态（6次检查 = 60秒）
        if check_count % 6 == 0:
            elapsed_minutes = check_count // 6
            status = f"运行时间: {elapsed_minutes}分钟 | 服务器={'运行' if server_alive else '停止'}, 活跃Workers={alive_count}/3"
            info(f"{status}\n")
            
            # 每5分钟显示一次训练进度（从日志中提取）
            if elapsed_minutes % 5 == 0 and elapsed_minutes > 0:
                info(f"=== {elapsed_minutes}分钟进度检查 ===\n")
                # 安全显示日志 - 过滤二进制数据，仅显示统计信息
                try:
                    # 统计服务器完成的传输数量
                    server_success_count = server.cmd(f"grep -c 'successfully sent all the tensor data' {project_dir}/logs/server.log").strip()
                    info(f"服务器状态: 已完成 {server_success_count} 个张量传输\n")
                except Exception as e:
                    info(f"服务器日志统计失败: {e}\n")
                
                try:
                    # 统计worker完成的任务数量
                    worker0_success_count = worker0.cmd(f"grep -c 'Successfully completed' {project_dir}/logs/worker0.log").strip()
                    info(f"Worker0(独立host): 已完成 {worker0_success_count} 个任务\n")
                except Exception as e:
                    info(f"Worker0日志统计失败: {e}\n")
                
                try:
                    worker1_success_count = server.cmd(f"grep -c 'Successfully completed' {project_dir}/logs/worker1.log").strip()
                    info(f"Worker1(本地): 已完成 {worker1_success_count} 个任务\n")
                except Exception as e:
                    info(f"Worker1日志统计失败: {e}\n")
        
        # 如果所有Workers都停止了，说明训练完成（服务器可能仍在清理）
        if alive_count == 0:
            total_minutes = check_count // 6
            info(f"🎉 训练完成! 所有Workers已结束，总用时约: {total_minutes}分钟\n")
            break
        
        # 如果只有服务器停止但worker还在运行，可能有问题
        if not server_alive and alive_count > 0:
            info("⚠️ 服务器已停止但Worker仍在运行，可能出现异常\n")
            break
        
        # 安全检查：如果运行超过2小时（720次检查），强制停止
        if check_count > 720:
            info("⚠️ 运行时间超过2小时，强制停止监控\n")
            break
    
    # 6. 显示最终统计结果 (避免二进制数据)
    info("6. 显示训练结果统计...\n")
    
    # 统计训练完成情况
    try:
        server_transfers = server.cmd(f"grep -c 'successfully sent all the tensor data' {project_dir}/logs/server.log").strip()
        info(f"✅ 服务器完成张量传输: {server_transfers} 次\n")
    except Exception as e:
        info(f"❌ 无法统计服务器传输次数: {e}\n")
    
    try:
        worker0_tasks = worker0.cmd(f"grep -c 'Successfully completed' {project_dir}/logs/worker0.log").strip()
        info(f"✅ Worker0 (独立host) 完成任务: {worker0_tasks} 次\n")
    except Exception as e:
        info(f"❌ 无法统计Worker0任务次数: {e}\n")
    
    try:
        worker1_tasks = server.cmd(f"grep -c 'Successfully completed' {project_dir}/logs/worker1.log").strip()
        info(f"✅ Worker1 (server本地) 完成任务: {worker1_tasks} 次\n")
    except Exception as e:
        info(f"❌ 无法统计Worker1任务次数: {e}\n")
    
    try:
        worker2_tasks = server.cmd(f"grep -c 'Successfully completed' {project_dir}/logs/worker2.log").strip()
        info(f"✅ Worker2 (server本地) 完成任务: {worker2_tasks} 次\n")
    except Exception as e:
        info(f"❌ 无法统计Worker2任务次数: {e}\n")
    
    # 显示文件大小统计
    info("=== 日志文件大小 ===\n")
    try:
        log_sizes = server.cmd(f"ls -lh {project_dir}/logs/")
        info(f"{log_sizes}\n")
    except Exception as e:
        info(f"无法显示日志文件大小: {e}\n")
    
    # 7. 显示日志文件位置
    info("7. 日志文件位置:\n")
    info(f"服务器日志: {project_dir}/logs/server.log\n")
    for i in range(3):
        info(f"Worker{i} 日志: {project_dir}/logs/worker{i}.log\n")
    
    # 8. 清理进程
    info("8. 清理进程...\n")
    server.cmd("pkill -f server_multithreading")
    server.cmd("pkill -f worker_multithreading")  # 清理server本地的workers
    worker0.cmd("pkill -f worker_multithreading")  # 清理独立host的worker
    
    # 停止网络
    net.stop()
    info("✅ 测试完成!\n")

if __name__ == "__main__":
    setLogLevel('info')
    test_mlt_training() 