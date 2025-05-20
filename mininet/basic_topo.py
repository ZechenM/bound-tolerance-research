#!/usr/bin/env python

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel, info
from mininet.cli import CLI

import time
import argparse
import json

class SimpleTopology(Topo):
    """简单的服务器-工作节点拓扑：
    
    server --- switch --- worker1
                   |
                   |--- worker2
                   |
                   |--- worker3
    """
    def build(self, bw=100, delay='1ms', loss=0):
        # 添加主机
        server = self.addHost('server')
        worker1 = self.addHost('worker1')
        worker2 = self.addHost('worker2')
        worker3 = self.addHost('worker3')
        
        # 添加交换机
        switch = self.addSwitch('s1')
        
        # 连接服务器到交换机
        self.addLink(server, switch, bw=bw, delay=delay, loss=loss)
        
        # 连接工作节点到交换机
        self.addLink(worker1, switch, bw=bw, delay=delay, loss=loss)
        self.addLink(worker2, switch, bw=bw, delay=delay, loss=loss)
        self.addLink(worker3, switch, bw=bw, delay=delay, loss=loss)

def run_iperf_test(net, server, worker, duration=15):
    """运行 iperf 测试并返回结果"""
    try:
        # 启动服务器
        info(f"*** 在 {server.name} 上启动 iperf 服务器\n")
        server.cmd('iperf -s &')
        time.sleep(2)  # 给服务器一些启动时间
        
        # 检查服务器是否正在运行
        server_status = server.cmd('ps aux | grep iperf | grep -v grep')
        info(f"*** 服务器状态: {server_status}\n")
        
        # 检查网络连接
        info(f"*** 检查 {worker.name} 到 {server.name} 的连接\n")
        ping_result = worker.cmd(f'ping -c 1 {server.IP()}')
        info(f"*** Ping 结果: {ping_result}\n")
        
        # 运行客户端测试，使用更大的窗口大小和更长的测试时间
        info(f"*** 从 {worker.name} 运行 iperf 客户端测试\n")
        result = worker.cmd(f'iperf -c {server.IP()} -t {duration} -i 1 -w 1M -M 1500')
        info(f"*** Iperf 原始输出: {result}\n")
        
        # 清理服务器进程
        server.cmd('kill %iperf')
        
        return result
    except Exception as e:
        info(f"*** 错误: 测试过程中出现异常: {str(e)}\n")
        return f"Error: {str(e)}"

def perfTest(bw=100, delay='1ms', loss=0, only_net=False):
    """创建网络并运行简单的性能测试"""
    topo = SimpleTopology(bw=bw, delay=delay, loss=loss)
    net = Mininet(topo=topo, link=TCLink)
    net.start()
    
    # 打印节点连接
    info("*** 转储主机连接\n")
    dumpNodeConnections(net.hosts)
    
    # 打印配置信息
    info(f"*** 网络参数: 带宽={bw}Mbps, 延迟={delay}, 丢包率={loss}%\n")
    
    if only_net:
        return net
    
    # 获取节点
    server = net.get('server')
    worker1 = net.get('worker1')
    worker2 = net.get('worker2')
    worker3 = net.get('worker3')
    
    # 运行测试并收集结果
    results = []
    for worker in [worker1, worker2, worker3]:
        info(f"\n*** 开始测试 {worker.name} -> server ***\n")
        result = run_iperf_test(net, server, worker)
        info(f"*** {worker.name} 测试结果: {result}\n")
        results.append(result)
        
        # 在测试之间添加延迟
        time.sleep(3)
    
    # 停止网络
    net.stop()
    
    return results

if __name__ == '__main__':
    setLogLevel('info')
    
    parser = argparse.ArgumentParser(description='测试网络丢包率与吞吐量关系')
    parser.add_argument('--bw', type=int, default=100, help='带宽 (Mbps)')
    parser.add_argument('--delay', type=str, default='1ms', help='延迟 (例如 5ms)')
    parser.add_argument('--loss', type=float, default=0, help='丢包率 (%)')
    parser.add_argument('--only-net', action='store_true', help='只启动网络，不运行 iperf 测试')
    
    args = parser.parse_args()
    
    results = perfTest(bw=args.bw, delay=args.delay, loss=args.loss, only_net=args.only_net)
    
    # 如果不是只启动网络，则输出 JSON 格式的结果
    if not args.only_net:
        print(json.dumps(results))