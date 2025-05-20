#!/usr/bin/env python

import os
import sys
import time
import subprocess
import argparse
import numpy as np
import csv
import json
from datetime import datetime
import re

# 不同的丢包率配置
LOSS_RATES = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

def parse_iperf_output(output):
    """解析 iperf 输出，提取总吞吐量（最后一行）"""
    lines = output.strip().split('\n')
    for line in reversed(lines):
        if 'Mbits/sec' in line:
            match = re.search(r'(\d+\.?\d*)\s+Mbits/sec', line)
            if match:
                return float(match.group(1))
    return 0.0

def run_test_all(bandwidth=100, delay='5ms', iterations=5, data_size=1048576):
    """
    运行一系列测试，对每个丢包率进行多次测试，并记录结果
    
    Args:
        bandwidth: 链路带宽 (Mbps)
        delay: 链路延迟
        iterations: 每个丢包率测试的次数
        data_size: 每次发送的数据大小 (bytes)
    """
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 创建结果CSV文件
    results_file = os.path.join(results_dir, "throughput_results.csv")
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['loss_rate', 'iteration', 'worker_id', 'throughput']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        try:
            # 批量复制所有相关文件到容器根目录
            print("复制所有相关文件到容器根目录 ...")
            for fname in ["basic_topo.py", "simple_server.py", "simple_worker.py"]:
                subprocess.run(["docker", "cp", fname, f"mininet:/{fname}"], check=True)
            
            # 对每个丢包率进行测试
            for loss_rate in LOSS_RATES:
                print(f"\n======= 测试丢包率 {loss_rate}% =======")
                
                for iteration in range(1, iterations + 1):
                    print(f"\n--- 迭代 {iteration}/{iterations} ---")
                    
                    try:
                        # 运行测试
                        print(f"运行测试 (带宽={bandwidth}Mbps, 延迟={delay}, 丢包率={loss_rate}%)")
                        cmd = [
                            "docker", "exec", "mininet", "python3", "/basic_topo.py",
                            "--bw", str(bandwidth),
                            "--delay", delay,
                            "--loss", str(loss_rate)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            print(f"警告: 测试运行失败")
                            print(result.stderr)
                            continue
                        
                        # 解析 JSON 结果
                        try:
                            results = json.loads(result.stdout)
                            for worker_id, output in enumerate(results):
                                throughput = parse_iperf_output(output)
                                
                                # 记录结果
                                writer.writerow({
                                    'loss_rate': loss_rate,
                                    'iteration': iteration,
                                    'worker_id': worker_id,
                                    'throughput': throughput
                                })
                                csvfile.flush()  # 确保写入磁盘
                                
                                print(f"工作节点 {worker_id} 吞吐量: {throughput:.2f} Mbps")
                        except json.JSONDecodeError:
                            print("警告: 无法解析测试结果")
                            print(result.stdout)
                        
                        # 等待一段时间再进行下一次迭代
                        time.sleep(2)
                        
                    except Exception as e:
                        print(f"测试过程中出现错误: {str(e)}")
                        continue
            
        except Exception as e:
            print(f"测试过程中出现错误: {str(e)}")
    
    print(f"\n测试完成。结果已保存到 {results_file}")
    print(f"使用以下命令生成结果图表：")
    print(f"python plot_results.py {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行一系列丢包率测试")
    parser.add_argument("--bw", type=int, default=100, help="带宽 (Mbps)")
    parser.add_argument("--delay", type=str, default="5ms", help="延迟")
    parser.add_argument("--iterations", type=int, default=3, help="每个丢包率测试的次数")
    parser.add_argument("--size", type=int, default=1048576, help="每次发送的数据大小 (bytes)")
    
    args = parser.parse_args()
    
    run_test_all(
        bandwidth=args.bw,
        delay=args.delay,
        iterations=args.iterations,
        data_size=args.size
    ) 