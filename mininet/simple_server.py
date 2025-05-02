#!/usr/bin/env python

import socket
import time
import threading
import argparse
import struct
import pickle
import numpy as np

class SimpleServer:
    def __init__(self, port=5000, host='0.0.0.0', num_workers=3):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.server_socket = None
        self.connections = []
        self.throughput_data = []
        self.running = True
        
    def start_server(self):
        """启动服务器并监听连接"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_workers)
        print(f"服务器启动，监听 {self.host}:{self.port}...")
        
        # 等待所有工作节点连接
        for i in range(self.num_workers):
            conn, addr = self.server_socket.accept()
            self.connections.append((conn, addr))
            print(f"工作节点 {i+1} 已连接: {addr}")
            
            # 为每个连接启动一个处理线程
            thread = threading.Thread(target=self.handle_worker, args=(conn, addr, i))
            thread.daemon = True
            thread.start()
        
        # 主线程等待用户输入以停止服务器
        try:
            while self.running:
                cmd = input("输入 'q' 停止服务器: ")
                if cmd.lower() == 'q':
                    self.running = False
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_server()
    
    def handle_worker(self, conn, addr, worker_id):
        """处理来自工作节点的通信"""
        print(f"开始处理来自工作节点 {worker_id+1} 的通信")
        try:
            while self.running:
                # 接收数据大小
                size_data = conn.recv(4)
                if not size_data:
                    break
                    
                data_size = struct.unpack("!I", size_data)[0]
                
                # 接收实际数据
                start_time = time.time()
                data = b""
                bytes_received = 0
                
                while bytes_received < data_size and self.running:
                    chunk = conn.recv(min(4096, data_size - bytes_received))
                    if not chunk:
                        break
                    data += chunk
                    bytes_received += len(chunk)
                
                if bytes_received != data_size:
                    print(f"警告: 只收到 {bytes_received}/{data_size} 字节")
                    break
                
                end_time = time.time()
                
                # 计算吞吐量 (Mbps)
                duration = end_time - start_time
                if duration > 0:
                    throughput = (bytes_received * 8) / (duration * 1000000)  # 转换为Mbps
                    self.throughput_data.append((worker_id, throughput, bytes_received, duration))
                    print(f"工作节点 {worker_id+1} 吞吐量: {throughput:.2f} Mbps (收到 {bytes_received/1024:.2f} KB 用时 {duration:.4f} 秒)")
                
                # 发送确认
                conn.sendall(b"ACK")
                
        except Exception as e:
            print(f"处理工作节点 {worker_id+1} 通信时出错: {e}")
        finally:
            conn.close()
            print(f"工作节点 {worker_id+1} 连接已关闭")
    
    def stop_server(self):
        """停止服务器并输出统计数据"""
        print("\n停止服务器...")
        self.running = False
        
        # 关闭所有连接
        for conn, _ in self.connections:
            try:
                conn.close()
            except:
                pass
        
        # 关闭服务器套接字
        if self.server_socket:
            self.server_socket.close()
        
        # 输出吞吐量统计
        if self.throughput_data:
            throughputs = [t[1] for t in self.throughput_data]
            print("\n吞吐量统计:")
            print(f"  平均吞吐量: {np.mean(throughputs):.2f} Mbps")
            print(f"  最大吞吐量: {np.max(throughputs):.2f} Mbps")
            print(f"  最小吞吐量: {np.min(throughputs):.2f} Mbps")
            print(f"  测量总数: {len(throughputs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简单的服务器用于测试吞吐量")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--workers", type=int, default=3, help="工作节点数量")
    
    args = parser.parse_args()
    
    server = SimpleServer(port=args.port, num_workers=args.workers)
    server.start_server() 