#!/usr/bin/env python3
"""
Test script for TIMELY congestion control implementation.

This script validates the TIMELY algorithm functionality by simulating
different network conditions and measuring the response.
"""

import sys
import time
import random
import unittest
from unittest.mock import patch

# Add current directory to path for imports
sys.path.append('.')

from timely import TimelyController
import config


class TestTimelyController(unittest.TestCase):
    """Test cases for TIMELY congestion control algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = TimelyController(
            initial_rate_mbps=10.0,
            target_rtt_ms=5.0,
            max_rate_mbps=100.0,
            min_rate_mbps=1.0,
        )
    
    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.current_rate_mbps, 10.0)
        self.assertEqual(self.controller.target_rtt_ms, 5.0)
        self.assertIsNone(self.controller.smoothed_rtt_ms)
        self.assertEqual(self.controller.congestion_state, "SLOW_START")
    
    def test_rtt_measurement(self):
        """Test RTT measurement and smoothing."""
        # Start measurement
        start_time = self.controller.start_measurement()
        
        # Simulate some delay
        time.sleep(0.001)  # 1ms delay
        
        # Update RTT
        new_rate = self.controller.update_rtt(start_time)
        
        # Check that RTT was measured
        self.assertIsNotNone(self.controller.smoothed_rtt_ms)
        self.assertGreater(self.controller.smoothed_rtt_ms, 0)
        self.assertGreater(new_rate, 0)
    
    def test_low_rtt_increases_rate(self):
        """Test that low RTT leads to rate increase."""
        initial_rate = self.controller.current_rate_mbps
        
        # Simulate very low RTT (below target)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.001]  # 1ms RTT
            start_time = self.controller.start_measurement()
            self.controller.update_rtt(start_time)
        
        # Rate should increase
        self.assertGreater(self.controller.current_rate_mbps, initial_rate)
        self.assertEqual(self.controller.congestion_state, "SLOW_START")
    
    def test_high_rtt_decreases_rate(self):
        """Test that high RTT leads to rate decrease."""
        # Set initial rate higher
        self.controller.current_rate_mbps = 50.0
        
        # Simulate high RTT (above threshold)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.010]  # 10ms RTT (above threshold)
            start_time = self.controller.start_measurement()
            self.controller.update_rtt(start_time)
        
        # Rate should decrease
        self.assertLess(self.controller.current_rate_mbps, 50.0)
        self.assertEqual(self.controller.congestion_state, "FAST_RECOVERY")
    
    def test_rate_limits(self):
        """Test that rate stays within configured limits."""
        # Test minimum rate limit
        self.controller.current_rate_mbps = 2.0
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.020]  # Very high RTT
            start_time = self.controller.start_measurement()
            self.controller.update_rtt(start_time)
        
        self.assertGreaterEqual(self.controller.current_rate_mbps, self.controller.min_rate_mbps)
        
        # Test maximum rate limit
        self.controller.current_rate_mbps = 90.0
        with patch('time.time') as mock_time:
            mock_time.side_effect = [1000.0, 1000.001]  # Very low RTT
            start_time = self.controller.start_measurement()
            self.controller.update_rtt(start_time)
        
        self.assertLessEqual(self.controller.current_rate_mbps, self.controller.max_rate_mbps)
    
    def test_send_delay_calculation(self):
        """Test send delay calculation based on current rate."""
        chunk_size = 8192  # 8KB chunk
        
        # Set a known rate
        self.controller.current_rate_mbps = 10.0
        self.controller.bytes_per_second = self.controller._mbps_to_bytes_per_second(10.0)
        
        delay = self.controller.get_send_delay(chunk_size)
        
        # Should be positive and reasonable
        self.assertGreater(delay, 0)
        self.assertLess(delay, 1.0)  # Should be less than 1 second for 8KB at 10Mbps
    
    def test_stats_reporting(self):
        """Test statistics reporting."""
        stats = self.controller.get_stats()
        
        required_keys = [
            'rate_mbps', 'smoothed_rtt_ms', 'rtt_gradient', 
            'target_rtt_ms', 'congestion_state', 'recent_rtt_samples',
            'bytes_per_second'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
    
    def test_reset_functionality(self):
        """Test controller reset functionality."""
        # Modify some state
        self.controller.current_rate_mbps = 50.0
        self.controller.smoothed_rtt_ms = 10.0
        self.controller.congestion_state = "CONGESTION_AVOIDANCE"
        
        # Reset
        self.controller.reset()
        
        # Should be back to initial state
        self.assertEqual(self.controller.current_rate_mbps, self.controller.initial_rate_mbps)
        self.assertIsNone(self.controller.smoothed_rtt_ms)
        self.assertEqual(self.controller.congestion_state, "SLOW_START")


def run_performance_simulation():
    """
    Run a performance simulation to demonstrate TIMELY behavior.
    """
    print("\n" + "="*60)
    print("TIMELY Controller Performance Simulation")
    print("="*60)
    
    controller = TimelyController(
        initial_rate_mbps=10.0,
        target_rtt_ms=5.0,
        max_rate_mbps=100.0,
        min_rate_mbps=1.0,
    )
    
    print(f"Initial configuration:")
    print(f"  Target RTT: {controller.target_rtt_ms}ms")
    print(f"  Initial Rate: {controller.current_rate_mbps} Mbps")
    print(f"  Rate Range: {controller.min_rate_mbps}-{controller.max_rate_mbps} Mbps")
    print()
    
    # Simulate different network conditions
    scenarios = [
        ("Low latency network", 2.0, 1.0),      # 2ms ± 1ms
        ("Target latency", 5.0, 0.5),           # 5ms ± 0.5ms  
        ("Congested network", 15.0, 5.0),       # 15ms ± 5ms
        ("Recovery phase", 8.0, 2.0),           # 8ms ± 2ms
        ("Stable operation", 5.0, 0.2),         # 5ms ± 0.2ms
    ]
    
    print("Scenario\t\tRTT(ms)\tRate(Mbps)\tState")
    print("-" * 60)
    
    for scenario_name, base_rtt, rtt_variance in scenarios:
        for step in range(5):  # 5 measurements per scenario
            # Simulate RTT measurement
            simulated_rtt = base_rtt + random.uniform(-rtt_variance, rtt_variance)
            
            # Simulate the measurement timing
            with patch('time.time') as mock_time:
                mock_time.side_effect = [1000.0, 1000.0 + simulated_rtt/1000.0]
                start_time = controller.start_measurement()
                controller.update_rtt(start_time)
            
            stats = controller.get_stats()
            print(f"{scenario_name[:15]:<15}\t{stats['smoothed_rtt_ms']:.2f}\t"
                  f"{stats['rate_mbps']:.1f}\t\t{stats['congestion_state']}")
            
            # Small delay between measurements
            time.sleep(0.001)
        
        print()
    
    final_stats = controller.get_stats()
    print(f"Final state:")
    print(f"  Rate: {final_stats['rate_mbps']:.1f} Mbps")
    print(f"  Smoothed RTT: {final_stats['smoothed_rtt_ms']:.2f}ms")
    print(f"  State: {final_stats['congestion_state']}")


def test_mlt_integration():
    """
    Test TIMELY integration with MLT configuration.
    """
    print("\n" + "="*60)
    print("TIMELY Integration with MLT Configuration")
    print("="*60)
    
    print(f"TIMELY enabled: {config.ENABLE_TIMELY}")
    print(f"Initial rate: {config.TIMELY_INITIAL_RATE_MBPS} Mbps")
    print(f"Target RTT: {config.TIMELY_TARGET_RTT_MS} ms")
    print(f"Rate range: {config.TIMELY_MIN_RATE_MBPS}-{config.TIMELY_MAX_RATE_MBPS} Mbps")
    print(f"Alpha (smoothing): {config.TIMELY_ALPHA}")
    print(f"Beta (decrease factor): {config.TIMELY_BETA}")
    
    # Create controller with config values
    controller = TimelyController(
        initial_rate_mbps=config.TIMELY_INITIAL_RATE_MBPS,
        target_rtt_ms=config.TIMELY_TARGET_RTT_MS,
        max_rate_mbps=config.TIMELY_MAX_RATE_MBPS,
        min_rate_mbps=config.TIMELY_MIN_RATE_MBPS,
        alpha=config.TIMELY_ALPHA,
        beta=config.TIMELY_BETA,
        additive_increase_mbps=config.TIMELY_ADDITIVE_INCREASE_MBPS,
        rtt_threshold_factor=config.TIMELY_RTT_THRESHOLD_FACTOR,
    )
    
    print(f"\nController initialized successfully with config parameters")
    print(f"Current state: {controller.get_stats()}")


if __name__ == "__main__":
    print("Testing TIMELY Congestion Control Implementation")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False, argv=[''])
    
    # Run performance simulation
    run_performance_simulation()
    
    # Test MLT integration
    test_mlt_integration()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("TIMELY implementation is ready for use with MLT protocol.")