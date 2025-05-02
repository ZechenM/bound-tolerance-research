#!/usr/bin/env python

import socket
import time
import argparse
import struct
import numpy as np
import random

class SimpleWorker:
    def __init__(self, server_host, server_port=5000, worker_id=0, data_size=1048576, num_iterations=10):
        """简单的工作节点客户端

        Args:
            server_host: 服务器主机名或IP
            server_port: 服务器端口
            worker_id: 工作节点ID
            data_size: 每次发送的数据大小（字节）
            num_iterations: 发送数据的次数
        """
        self.server_host = server_host
        self.server_port = server_port
        self.worker_id = worker_id
        self.data_size = data_size
        self.num_iterations = num_iterations
        
    def run(self):
        """运行工作节点，连接到服务器并发送数据"""
        print(f"工作节点 {self.worker_id} 启动，连接到 {self.server_host}:{self.server_port}")
        
        try:
            # 创建套接字并连接到服务器
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_host, self.server_port))
            
            print(f"已连接到服务器，开始发送数据")
            
            throughputs = []
            for i in range(self.num_iterations):
                # 创建随机数据
                data = np.random.bytes(self.data_size)
                
                # 发送数据大小
                sock.sendall(struct.pack("!I", len(data)))
                
                # 发送数据并计时
                start_time = time.time()
                sock.sendall(data)
                
                # 等待服务器确认
                ack = sock.recv(3)
                if ack != b"ACK":
                    print(f"警告: 预期收到ACK，但收到: {ack}")
                
                end_time = time.time()
                
                # 计算吞吐量 (Mbps)
                duration = end_time - start_time
                if duration > 0:
                    throughput = (self.data_size * 8) / (duration * 1000000)  # 转换为Mbps
                    throughputs.append(throughput)
                    print(f"迭代 {i+1}/{self.num_iterations} 吞吐量: {throughput:.2f} Mbps (发送 {self.data_size/1024:.2f} KB 用时 {duration:.4f} 秒)")
                
                # 随机等待一段时间，模拟工作
                time.sleep(random.uniform(0.1, 0.5))
            
            # 输出统计信息
            if throughputs:
                print("\n吞吐量统计:")
                print(f"  平均吞吐量: {np.mean(throughputs):.2f} Mbps")
                print(f"  最大吞吐量: {np.max(throughputs):.2f} Mbps")
                print(f"  最小吞吐量: {np.min(throughputs):.2f} Mbps")
            
        except Exception as e:
            print(f"连接或发送数据时出错: {e}")
        finally:
            try:
                sock.close()
            except:
                pass
            print("工作节点已关闭")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简单的工作节点客户端用于测试吞吐量")
    parser.add_argument("--server", default="localhost", help="服务器主机名或IP")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--id", type=int, default=0, help="工作节点ID")
    parser.add_argument("--size", type=int, default=1048576, help="每次发送的数据大小（字节）")
    parser.add_argument("--iterations", type=int, default=10, help="发送数据的次数")
    
    args = parser.parse_args()
    
    worker = SimpleWorker(
        server_host=args.server,
        server_port=args.port,
        worker_id=args.id,
        data_size=args.size,
        num_iterations=args.iterations
    )
    worker.run() 