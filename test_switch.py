#!/usr/bin/env python3
"""
测试Mininet中的交换机是否能正常工作
"""

from mininet.net import Mininet
from mininet.log import info, setLogLevel

def test_switch():
    """测试简单的交换机网络"""
    
    net = Mininet()
    
    # 添加主机
    h1 = net.addHost('h1', ip='10.0.0.1/24')
    h2 = net.addHost('h2', ip='10.0.0.2/24')
    h3 = net.addHost('h3', ip='10.0.0.3/24')
    
    # 添加交换机
    s1 = net.addSwitch('s1')
    
    # 连接到交换机
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)
    
    info("*** 启动网络\n")
    net.start()
    
    info("*** 测试连通性\n")
    result = net.pingAll()
    
    if result == 0:
        info("✅ 交换机网络工作正常\n")
    else:
        info(f"⚠️ 交换机网络有问题，丢包率: {result}%\n")
    
    # 测试特定连接
    info("*** 测试h1到h2连接\n")
    result = net.ping([h1, h2])
    info(f"h1-h2 ping结果: {result}% 丢包\n")
    
    net.stop()
    return result == 0

if __name__ == "__main__":
    setLogLevel('info')
    success = test_switch()
    if success:
        info("交换机测试成功，可以使用交换机拓扑\n")
    else:
        info("交换机测试失败，需要使用其他方案\n")