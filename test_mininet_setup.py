#!/usr/bin/env python3
"""
Mininet环境测试脚本
用于验证Mininet安装和配置是否正确
"""

import os
import sys
import subprocess
import time

def test_mininet_installation():
    """测试Mininet是否已安装"""
    print("=== 测试Mininet安装 ===")
    
    try:
        # 检查mn命令
        result = subprocess.run(['mn', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ Mininet已安装")
            print(f"  版本: {result.stdout.strip()}")
            return True
        else:
            print("✗ Mininet安装有问题")
            return False
    except FileNotFoundError:
        print("✗ Mininet未安装")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Mininet命令超时")
        return False

def test_python_dependencies():
    """测试Python依赖包"""
    print("\n=== 测试Python依赖 ===")
    
    required_packages = [
        'torch', 'torchvision', 'transformers', 
        'sklearn', 'numpy', 'mininet'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少的包: {', '.join(missing_packages)}")
        print("请安装: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_data_files():
    """测试数据文件是否存在"""
    print("\n=== 测试数据文件 ===")
    
    data_dir = "data/cifar10_splits"
    required_files = [
        "train_0.pth",
        "train_1.pth", 
        "train_2.pth",
        "test.pth"
    ]
    
    if not os.path.exists(data_dir):
        print(f"✗ 数据目录不存在: {data_dir}")
        print("请运行: python prepare_data.py")
        return False
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - 不存在")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n缺少的文件: {', '.join(missing_files)}")
        print("请运行: python prepare_data.py")
        return False
    
    return True

def test_simple_topology():
    """测试简单的Mininet拓扑"""
    print("\n=== 测试Mininet拓扑 ===")
    
    try:
        # 创建简单的拓扑
        from mininet.net import Mininet
        from mininet.topo import Topo
        from mininet.node import OVSKernelSwitch, RemoteController
        
        class SimpleTopo(Topo):
            def __init__(self):
                Topo.__init__(self)
                h1 = self.addHost('h1')
                h2 = self.addHost('h2')
                s1 = self.addSwitch('s1')
                self.addLink(h1, s1)
                self.addLink(h2, s1)
        
        # 创建网络
        net = Mininet(
            topo=SimpleTopo(),
            switch=OVSKernelSwitch,
            controller=RemoteController
        )
        
        # 启动网络
        net.start()
        print("✓ Mininet网络启动成功")
        
        # 测试连通性
        result = net.pingAll()
        if result == 0:
            print("✓ 网络连通性测试通过")
        else:
            print("✗ 网络连通性测试失败")
        
        # 停止网络
        net.stop()
        print("✓ Mininet网络停止成功")
        
        return True
        
    except Exception as e:
        print(f"✗ Mininet拓扑测试失败: {e}")
        return False

def test_script_files():
    """测试脚本文件是否存在"""
    print("\n=== 测试脚本文件 ===")
    
    required_scripts = [
        "server_multithreading.py",
        "worker_multithreading.py",
        "mininet/multithreading_topo.py"
    ]
    
    missing_scripts = []
    
    for script in required_scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            print(f"✗ {script} - 不存在")
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"\n缺少的脚本: {', '.join(missing_scripts)}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("Mininet环境测试")
    print("=" * 50)
    
    tests = [
        ("Mininet安装", test_mininet_installation),
        ("Python依赖", test_python_dependencies),
        ("数据文件", test_data_files),
        ("脚本文件", test_script_files),
        ("Mininet拓扑", test_simple_topology),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试出错: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境配置正确。")
        print("可以运行: sudo python3 mininet/multithreading_topo.py")
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败。")
        print("请根据上述错误信息修复问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 