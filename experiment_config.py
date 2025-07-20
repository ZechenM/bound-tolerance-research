# 实验配置文件
# 用于控制MLT协议与TCP协议的比较实验

# 协议选择: "TCP" 或 "MLT"
PROTOCOL = "MLT"  # 可以改为 "TCP" 来测试TCP协议

# 网络配置
NETWORK_CONFIG = {
    "bandwidth": 1000,      # 带宽 (Mbps)
    "delay": "1ms",         # 延迟
    "loss": 0,              # 丢包率 (%)
    "queue_size": 10000,    # 队列大小
}

# MLT协议特定配置
MLT_CONFIG = {
    "loss_tolerance": 0,    # 损失容忍度
    "chunk_size": 8192,     # 数据块大小 (字节)
}

# 实验参数
EXPERIMENT_CONFIG = {
    "num_workers": 3,       # 工作节点数量
    "server_port": 9999,    # 服务器端口
    "training_epochs": 3,   # 训练轮数
    "batch_size": 32,       # 批次大小
    "learning_rate": 0.001, # 学习率
}

# 日志配置
LOGGING_CONFIG = {
    "log_dir": "logs",      # 日志目录
    "log_level": "INFO",    # 日志级别
    "save_logs": True,      # 是否保存日志
}

def get_config():
    """获取当前实验配置"""
    return {
        "protocol": PROTOCOL,
        "network": NETWORK_CONFIG,
        "mlt": MLT_CONFIG,
        "experiment": EXPERIMENT_CONFIG,
        "logging": LOGGING_CONFIG,
    }

def print_config():
    """打印当前配置"""
    config = get_config()
    print("=== 当前实验配置 ===")
    print(f"协议: {config['protocol']}")
    print(f"网络配置: {config['network']}")
    print(f"MLT配置: {config['mlt']}")
    print(f"实验参数: {config['experiment']}")
    print("==================")

if __name__ == "__main__":
    print_config() 