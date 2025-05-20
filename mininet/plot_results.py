#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse  # 添加argparse模块导入

def plot_results(results_dir, use_log_scale=False):
    # Read data
    csv_path = os.path.join(results_dir, 'throughput_results.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Group by loss rate and calculate average throughput
    grouped = df.groupby('loss_rate')['throughput'].mean()

    # Calculate performance percentage (relative to max throughput)
    max_throughput = grouped.max()
    performance_percentage = (grouped / max_throughput) * 100

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(performance_percentage.index * 100, performance_percentage.values, marker='o', linestyle='-')
    
    # 根据参数决定是否使用对数刻度
    if use_log_scale:
        plt.xscale('log')  # 设置 x 轴为对数刻度
        plt.xlabel('Loss Rate (%) - Log Scale')  # 更新标签以反映对数刻度
        plt.title('Performance Degradation vs Loss Rate (Log Scale)')
        plt.grid(True, which="both", ls="-")  # 添加主要和次要网格线
        plt.minorticks_on()
        plt.grid(True, which="minor", ls="--", alpha=0.2)
    else:
        plt.xlabel('Loss Rate (%)')
        plt.title('Performance Degradation vs Loss Rate')
        plt.grid(True)
        
    plt.ylabel('Performance (% of Max Throughput)')
    plt.tight_layout()
    
    # Save plot in the results directory
    if use_log_scale:
        plot_path = os.path.join(results_dir, 'throughput_vs_loss_log.png')
    else:
        plot_path = os.path.join(results_dir, 'throughput_vs_loss.png')
    plt.savefig(plot_path)
    plt.show()

    # Print performance degradation table
    print("\nPerformance Degradation Table:")
    print("Loss Rate (%) | Performance (%) | Degradation (%)")
    print("-" * 50)
    for loss_rate, perf in performance_percentage.items():
        degradation = 100 - perf
        print(f"{loss_rate*100:11.2f} | {perf:13.2f} | {degradation:13.2f}")

if __name__ == "__main__":
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description='绘制丢包率与吞吐量关系图')
    parser.add_argument('results_dir', help='测试结果目录路径')
    parser.add_argument('--log', '-l', action='store_true', help='使用对数刻度绘制X轴')
    args = parser.parse_args()
    
    plot_results(args.results_dir, use_log_scale=args.log) 