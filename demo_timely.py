#!/usr/bin/env python3
"""
Comprehensive demonstration of TIMELY congestion control algorithm.

This script shows how TIMELY adapts to different network conditions
and provides performance improvements over fixed-rate UDP transmission.
"""

import sys
import time
import threading
from unittest.mock import patch

sys.path.append('.')

from timely import TimelyController
import config

def demonstrate_timely_adaptation():
    """Demonstrate TIMELY adaptation to various network conditions."""
    
    print("=" * 80)
    print("TIMELY Congestion Control Demonstration")
    print("=" * 80)
    
    controller = TimelyController(
        initial_rate_mbps=10.0,
        target_rtt_ms=5.0,
        max_rate_mbps=100.0,
        min_rate_mbps=1.0,
    )
    
    print(f"Configuration:")
    print(f"  Initial Rate: {controller.initial_rate_mbps} Mbps")
    print(f"  Target RTT: {controller.target_rtt_ms} ms")
    print(f"  Rate Range: {controller.min_rate_mbps} - {controller.max_rate_mbps} Mbps")
    print(f"  Alpha (smoothing): {controller.alpha}")
    print(f"  Beta (decrease): {controller.beta}")
    print()
    
    # Scenario 1: Excellent network conditions (low latency)
    print("Scenario 1: Excellent Network Conditions (Low Latency)")
    print("-" * 60)
    for i in range(5):
        with patch('time.time') as mock_time:
            # Simulate 1-2ms RTT (well below target)
            rtt_ms = 1.0 + i * 0.2
            mock_time.side_effect = [1000.0, 1000.0 + rtt_ms/1000.0]
            start_time = controller.start_measurement()
            controller.update_rtt(start_time)
        
        stats = controller.get_stats()
        print(f"  Step {i+1}: RTT={stats['smoothed_rtt_ms']:.2f}ms, "
              f"Rate={stats['rate_mbps']:.1f}Mbps, State={stats['congestion_state']}")
    
    print()
    
    # Scenario 2: Mild congestion (RTT near target)
    print("Scenario 2: Mild Congestion (RTT Near Target)")
    print("-" * 60)
    for i in range(5):
        with patch('time.time') as mock_time:
            # Simulate 4-6ms RTT (around target)
            rtt_ms = 4.0 + i * 0.5
            mock_time.side_effect = [1000.0, 1000.0 + rtt_ms/1000.0]
            start_time = controller.start_measurement()
            controller.update_rtt(start_time)
        
        stats = controller.get_stats()
        print(f"  Step {i+1}: RTT={stats['smoothed_rtt_ms']:.2f}ms, "
              f"Rate={stats['rate_mbps']:.1f}Mbps, State={stats['congestion_state']}")
    
    print()
    
    # Scenario 3: Heavy congestion (high RTT)
    print("Scenario 3: Heavy Congestion (High RTT)")
    print("-" * 60)
    for i in range(5):
        with patch('time.time') as mock_time:
            # Simulate 10-20ms RTT (well above threshold)
            rtt_ms = 10.0 + i * 2.0
            mock_time.side_effect = [1000.0, 1000.0 + rtt_ms/1000.0]
            start_time = controller.start_measurement()
            controller.update_rtt(start_time)
        
        stats = controller.get_stats()
        print(f"  Step {i+1}: RTT={stats['smoothed_rtt_ms']:.2f}ms, "
              f"Rate={stats['rate_mbps']:.1f}Mbps, State={stats['congestion_state']}")
    
    print()
    
    # Scenario 4: Recovery phase (RTT decreasing)
    print("Scenario 4: Recovery Phase (RTT Decreasing)")
    print("-" * 60)
    for i in range(5):
        with patch('time.time') as mock_time:
            # Simulate decreasing RTT from 15ms to 5ms
            rtt_ms = 15.0 - i * 2.5
            mock_time.side_effect = [1000.0, 1000.0 + rtt_ms/1000.0]
            start_time = controller.start_measurement()
            controller.update_rtt(start_time)
        
        stats = controller.get_stats()
        print(f"  Step {i+1}: RTT={stats['smoothed_rtt_ms']:.2f}ms, "
              f"Rate={stats['rate_mbps']:.1f}Mbps, State={stats['congestion_state']}")
    
    print()
    
    # Show final statistics
    final_stats = controller.get_stats()
    print("Final Controller State:")
    print(f"  Current Rate: {final_stats['rate_mbps']:.1f} Mbps")
    print(f"  Smoothed RTT: {final_stats['smoothed_rtt_ms']:.2f} ms")
    print(f"  RTT Gradient: {final_stats['rtt_gradient']:.2f} ms")
    print(f"  Congestion State: {final_stats['congestion_state']}")
    print(f"  Bytes per Second: {final_stats['bytes_per_second']:.0f}")

def demonstrate_performance_comparison():
    """Compare TIMELY-controlled vs fixed-rate transmission."""
    
    print("\n" + "=" * 80)
    print("Performance Comparison: TIMELY vs Fixed Rate")
    print("=" * 80)
    
    # TIMELY controller
    timely_controller = TimelyController(initial_rate_mbps=10.0, target_rtt_ms=5.0)
    
    # Fixed rate (no adaptation)
    fixed_rate_mbps = 10.0
    
    # Simulate network conditions over time
    network_conditions = [
        (1.0, "Excellent"),
        (2.0, "Good"),
        (5.0, "Target"),
        (8.0, "Congested"),
        (15.0, "Heavy Congestion"),
        (20.0, "Severe Congestion"),
        (10.0, "Recovery"),
        (5.0, "Stabilizing"),
        (3.0, "Good Again"),
        (1.5, "Excellent Again")
    ]
    
    print("Time\tNetwork RTT\tTIMELY Rate\tFixed Rate\tTIMELY Advantage")
    print("-" * 75)
    
    total_timely_throughput = 0
    total_fixed_throughput = 0
    
    for step, (network_rtt, condition) in enumerate(network_conditions):
        # Update TIMELY controller
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.0 + network_rtt/1000.0]
            start_time = timely_controller.start_measurement()
            timely_controller.update_rtt(start_time)
        
        timely_rate = timely_controller.get_current_rate_mbps()
        
        # Calculate advantage
        advantage = (timely_rate - fixed_rate_mbps) / fixed_rate_mbps * 100
        
        # Accumulate throughput (simplified metric)
        total_timely_throughput += timely_rate
        total_fixed_throughput += fixed_rate_mbps
        
        print(f"{step+1:2d}\t{network_rtt:6.1f}ms\t{timely_rate:8.1f}Mbps\t"
              f"{fixed_rate_mbps:8.1f}Mbps\t{advantage:+6.1f}%")
    
    # Summary
    overall_advantage = (total_timely_throughput - total_fixed_throughput) / total_fixed_throughput * 100
    print("-" * 75)
    print(f"Total Throughput: TIMELY={total_timely_throughput:.1f}, Fixed={total_fixed_throughput:.1f}")
    print(f"Overall TIMELY Advantage: {overall_advantage:+.1f}%")

def demonstrate_mlt_integration():
    """Show how TIMELY integrates with MLT configuration."""
    
    print("\n" + "=" * 80)
    print("TIMELY Integration with MLT Protocol")
    print("=" * 80)
    
    print("MLT Configuration Values:")
    print(f"  ENABLE_TIMELY = {config.ENABLE_TIMELY}")
    print(f"  TIMELY_INITIAL_RATE_MBPS = {config.TIMELY_INITIAL_RATE_MBPS}")
    print(f"  TIMELY_TARGET_RTT_MS = {config.TIMELY_TARGET_RTT_MS}")
    print(f"  TIMELY_MAX_RATE_MBPS = {config.TIMELY_MAX_RATE_MBPS}")
    print(f"  TIMELY_MIN_RATE_MBPS = {config.TIMELY_MIN_RATE_MBPS}")
    print(f"  TIMELY_ALPHA = {config.TIMELY_ALPHA}")
    print(f"  TIMELY_BETA = {config.TIMELY_BETA}")
    print(f"  CHUNK_SIZE = {config.CHUNK_SIZE} bytes")
    
    # Create controller with MLT config
    controller = TimelyController(
        initial_rate_mbps=config.TIMELY_INITIAL_RATE_MBPS,
        target_rtt_ms=config.TIMELY_TARGET_RTT_MS,
        max_rate_mbps=config.TIMELY_MAX_RATE_MBPS,
        min_rate_mbps=config.TIMELY_MIN_RATE_MBPS,
        alpha=config.TIMELY_ALPHA,
        beta=config.TIMELY_BETA,
    )
    
    print(f"\nChunk Sending Simulation:")
    print(f"Chunk size: {config.CHUNK_SIZE} bytes")
    
    # Simulate sending chunks with different rates
    rates_to_test = [1.0, 10.0, 50.0, 100.0]
    
    print("\nRate (Mbps)\tDelay per Chunk (ms)\tChunks per Second")
    print("-" * 55)
    
    for rate in rates_to_test:
        controller.current_rate_mbps = rate
        controller.bytes_per_second = controller._mbps_to_bytes_per_second(rate)
        
        delay_seconds = controller.get_send_delay(config.CHUNK_SIZE)
        delay_ms = delay_seconds * 1000
        chunks_per_second = 1.0 / delay_seconds if delay_seconds > 0 else float('inf')
        
        print(f"{rate:8.1f}\t{delay_ms:15.2f}\t{chunks_per_second:13.1f}")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_timely_adaptation()
    demonstrate_performance_comparison()
    demonstrate_mlt_integration()
    
    print("\n" + "=" * 80)
    print("TIMELY Implementation Successfully Demonstrated!")
    print("The algorithm provides adaptive congestion control based on RTT feedback,")
    print("improving performance and fairness in distributed ML communication.")
    print("=" * 80)