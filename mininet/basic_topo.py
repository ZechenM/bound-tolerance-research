#!/usr/bin/env python

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel, info
from mininet.cli import CLI

import time
import argparse

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

def perfTest(bw=100, delay='1ms', loss=0):
    """创建网络并运行简单的性能测试"""
    topo = SimpleTopology(bw=bw, delay=delay, loss=loss)
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink)
    net.start()
    
    # 打印节点连接
    info("*** 转储主机连接\n")
    dumpNodeConnections(net.hosts)
    
    # 打印配置信息
    info(f"*** 网络参数: 带宽={bw}Mbps, 延迟={delay}, 丢包率={loss}%\n")
    
    # 获取节点
    server = net.get('server')
    worker1 = net.get('worker1')
    worker2 = net.get('worker2')
    worker3 = net.get('worker3')
    
    # 启动服务器
    info("*** 启动服务器\n")
    server.cmd('iperf -s &')
    time.sleep(1)  # 给服务器一些启动时间
    
    # 从每个工作节点运行测试
    info("*** 从工作节点运行iperf测试\n")
    for worker in [worker1, worker2, worker3]:
        info(f"*** {worker.name} -> server: ")
        result = worker.cmd(f'iperf -c {server.IP()} -t 5')
        info(result)
    
    # 清理
    server.cmd('kill %iperf')
    
    # 进入CLI以便进一步交互测试
    CLI(net)
    
    # 停止网络
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    
    parser = argparse.ArgumentParser(description='测试网络丢包率与吞吐量关系')
    parser.add_argument('--bw', type=int, default=100, help='带宽 (Mbps)')
    parser.add_argument('--delay', type=str, default='1ms', help='延迟 (例如 5ms)')
    parser.add_argument('--loss', type=float, default=0, help='丢包率 (%)')
    
    args = parser.parse_args()
    
    perfTest(bw=args.bw, delay=args.delay, loss=args.loss) 