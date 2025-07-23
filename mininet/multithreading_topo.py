#!/usr/bin/env python3

import os
import time
import signal
import sys

from mininet.link import TCLink
from mininet.log import info, setLogLevel, error
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.topo import Topo
from mininet.cli import CLI

# 配置Python路径 - WSL适配
PYTHON = "python3"  # 在WSL中使用python3命令
PROJECT_DIR = os.getcwd()  # 使用当前工作目录

# 检测WSL环境
def is_wsl():
    """检测是否在WSL环境中运行"""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
    except:
        return False

# 获取虚拟环境Python路径
def get_python_path():
    """获取虚拟环境中的Python路径"""
    venv_python = f"{PROJECT_DIR}/venv/bin/python3"
    if os.path.exists(venv_python):
        return venv_python
    return PYTHON

# 导入实验配置
sys.path.append(PROJECT_DIR)
try:
    from experiment_config import get_config
    config = get_config()
except ImportError:
    # 如果配置文件不存在，使用默认配置
    config = {
        "protocol": "MLT",
        "network": {"bandwidth": 1000, "delay": "1ms", "loss": 0, "queue_size": 10000},
        "experiment": {"server_port": 9999}
    }

class MultithreadingTopo(Topo):
    """
    多线程MLT协议测试拓扑 (WSL适配版):
    3个工作节点和1个服务器通过1个交换机连接
    网络配置为高速低延迟，避免网络成为瓶颈
    """

    def __init__(self, **opts):
        "创建自定义拓扑"
        # 初始化拓扑
        Topo.__init__(self, **opts)

        # 添加节点 (主机: 服务器和工作节点; 交换机)
        server = self.addHost("server")
        worker0 = self.addHost("worker0")
        worker1 = self.addHost("worker1")
        worker2 = self.addHost("worker2")

        # 使用OVS Bridge交换机（WSL兼容）
        s1 = self.addSwitch("s1")

        # 定义链路特性 - WSL适配，使用最简单的配置
        if is_wsl():
            # 在WSL中不使用流量控制，避免qdisc问题
            link_opts = dict()
            info("*** 在WSL环境中使用默认链路配置（无流量控制）\n")
        else:
            # 在非WSL环境中使用完整配置
            link_opts = dict(
                bw=config["network"]["bandwidth"],           # 带宽
                delay=config["network"]["delay"],            # 延迟
                loss=config["network"]["loss"],              # 丢包率
                max_queue_size=config["network"]["queue_size"],  # 队列大小
                use_htb=True       # 使用HTB队列调度器
            )

        info("*** 添加高速网络链路:\n")
        info(f"* 所有链路配置: {link_opts}\n")

        # 添加工作节点到交换机的链路
        self.addLink(worker0, s1, **link_opts)
        self.addLink(worker1, s1, **link_opts)
        self.addLink(worker2, s1, **link_opts)

        # 添加服务器到交换机的链路
        self.addLink(server, s1, **link_opts)


def signal_handler(signum, frame):
    """处理Ctrl+C信号"""
    info("\n*** 收到中断信号，正在停止网络...\n")
    sys.exit(0)


def run_experiment():
    """
    创建网络，运行分布式ML应用程序，并启动CLI (WSL适配版)
    """
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 检测WSL环境
    if is_wsl():
        info("*** 检测到WSL环境\n")
        PYTHON_CMD = get_python_path()
        info(f"*** 使用Python路径: {PYTHON_CMD}\n")
    else:
        PYTHON_CMD = PYTHON
    
    # --- 配置 ---
    SERVER_SCRIPT = f"{PROJECT_DIR}/server_multithreading.py"
    WORKER_SCRIPT = f"{PROJECT_DIR}/worker_multithreading.py"
    SERVER_PORT = config["experiment"]["server_port"]  # 服务器端口

    # 定义日志路径
    LOG_DIR_BASE = f"{PROJECT_DIR}/logs"
    SERVER_LOG = f"{LOG_DIR_BASE}/server.log"
    WORKER0_LOG = f"{LOG_DIR_BASE}/worker0.log"
    WORKER1_LOG = f"{LOG_DIR_BASE}/worker1.log"
    WORKER2_LOG = f"{LOG_DIR_BASE}/worker2.log"
    # --- 配置结束 ---

    # 创建自定义拓扑实例
    topo = MultithreadingTopo()

    # 创建Mininet网络实例 - WSL适配，使用OVS Bridge
    net = Mininet(
        topo=topo,
        switch=OVSBridge,  # 使用OVS Bridge，不需要OpenFlow控制器
        controller=None,  # 明确指定不使用控制器
        link=TCLink,  # 使用TCLink进行流量控制
        autoSetMacs=True,
    )

    info("*** 启动网络\n")
    net.start()

    # 获取节点句柄
    server_node = net.get("server")
    worker0_node = net.get("worker0")
    worker1_node = net.get("worker1")
    worker2_node = net.get("worker2")

    # 获取服务器IP地址
    server_ip = server_node.IP()
    info(f"服务器IP地址: {server_ip}\n")
    info(f"服务器端口: {SERVER_PORT}\n")

    # 创建日志目录
    info(f"*** 在节点上创建日志目录 '{LOG_DIR_BASE}'...\n")
    for node in [server_node, worker0_node, worker1_node, worker2_node]:
        node.cmd(f"mkdir -p {LOG_DIR_BASE}")

    # 确保数据目录存在
    data_dir = f"{PROJECT_DIR}/data"
    for node in [server_node, worker0_node, worker1_node, worker2_node]:
        node.cmd(f"mkdir -p {data_dir}")

    # --- 启动分布式ML应用程序 ---
    info("*** 启动服务器应用程序...\n")
    
    # 创建服务器日志文件
    os.makedirs(os.path.dirname(SERVER_LOG), exist_ok=True)
    
    # 启动服务器 - WSL适配
    if is_wsl():
        # 在WSL中，需要激活虚拟环境
        server_cmd = f"cd {PROJECT_DIR} && source venv/bin/activate && {PYTHON_CMD} -u {SERVER_SCRIPT}"
    else:
        server_cmd = f"cd {PROJECT_DIR} && {PYTHON_CMD} -u {SERVER_SCRIPT}"
    
    info(f"在服务器上执行: {server_cmd}\n")
    
    server_proc = server_node.popen(
        server_cmd, 
        shell=True,  # 使用shell执行以支持source命令
        stdout=open(SERVER_LOG, "w"), 
        stderr=open(SERVER_LOG, "a")
    )

    # 等待服务器启动并绑定端口
    info("*** 等待服务器初始化...\n")
    time.sleep(5)  # 给服务器更多时间启动

    info("*** 启动工作节点应用程序...\n")
    
    # 创建工作节点日志文件
    for log_file in [WORKER0_LOG, WORKER1_LOG, WORKER2_LOG]:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 启动工作节点
    worker_processes = []
    
    # 设置环境变量
    env = os.environ.copy()
    env['SERVER_HOST'] = server_ip
    
    # Worker 0 - WSL适配
    if is_wsl():
        worker0_cmd = f"cd {PROJECT_DIR} && source venv/bin/activate && {PYTHON_CMD} -u {WORKER_SCRIPT} 0"
    else:
        worker0_cmd = f"cd {PROJECT_DIR} && {PYTHON_CMD} -u {WORKER_SCRIPT} 0"
    
    info(f"在worker0上执行: {worker0_cmd}\n")
    worker0_proc = worker0_node.popen(
        worker0_cmd, 
        shell=True,  # 使用shell执行以支持source命令
        stdout=open(WORKER0_LOG, "w"), 
        stderr=open(WORKER0_LOG, "a"),
        env=env
    )
    worker_processes.append(worker0_proc)

    # Worker 1 - WSL适配
    if is_wsl():
        worker1_cmd = f"cd {PROJECT_DIR} && source venv/bin/activate && {PYTHON_CMD} -u {WORKER_SCRIPT} 1"
    else:
        worker1_cmd = f"cd {PROJECT_DIR} && {PYTHON_CMD} -u {WORKER_SCRIPT} 1"
    
    info(f"在worker1上执行: {worker1_cmd}\n")
    worker1_proc = worker1_node.popen(
        worker1_cmd, 
        shell=True,  # 使用shell执行以支持source命令
        stdout=open(WORKER1_LOG, "w"), 
        stderr=open(WORKER1_LOG, "a"),
        env=env
    )
    worker_processes.append(worker1_proc)

    # Worker 2 - WSL适配
    if is_wsl():
        worker2_cmd = f"cd {PROJECT_DIR} && source venv/bin/activate && {PYTHON_CMD} -u {WORKER_SCRIPT} 2"
    else:
        worker2_cmd = f"cd {PROJECT_DIR} && {PYTHON_CMD} -u {WORKER_SCRIPT} 2"
    
    info(f"在worker2上执行: {worker2_cmd}\n")
    worker2_proc = worker2_node.popen(
        worker2_cmd, 
        shell=True,  # 使用shell执行以支持source命令
        stdout=open(WORKER2_LOG, "w"), 
        stderr=open(WORKER2_LOG, "a"),
        env=env
    )
    worker_processes.append(worker2_proc)

    # --- 应用程序启动结束 ---

    info("\n*** 运行基本连通性测试 (pingall)...\n")
    net.pingAll()

    info("\n*** 应用程序正在后台运行。\n")
    info(f"*** 服务器日志: {server_node.name}:{SERVER_LOG}\n")
    info(f"*** Worker 0 日志: {worker0_node.name}:{WORKER0_LOG}\n")
    info(f"*** Worker 1 日志: {worker1_node.name}:{WORKER1_LOG}\n")
    info(f"*** Worker 2 日志: {worker2_node.name}:{WORKER2_LOG}\n")
    info("*** 启动Mininet CLI。输入'exit'退出并停止网络。\n")
    info(f"*** 可以使用命令监控日志: tail -f {WORKER0_LOG}\n")

    try:
        # 启动CLI进行交互
        CLI(net)
    except KeyboardInterrupt:
        info("\n*** 收到键盘中断。正在停止网络...")
    finally:
        info("*** 停止网络\n")
        
        # 终止所有进程
        for proc in worker_processes:
            if proc.poll() is None:  # 如果进程还在运行
                proc.terminate()
                proc.wait()
        
        if server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait()
        
        net.stop()


if __name__ == "__main__":
    topos = {"multithreading": (lambda: MultithreadingTopo())}
    setLogLevel("info")
    run_experiment() 