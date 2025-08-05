"""
TIMELY congestion control algorithm implementation for UDP traffic.

The TIMELY algorithm uses RTT measurements as congestion signals to adjust
transmission rates dynamically. This implementation provides:
- RTT measurement and tracking
- Adaptive rate control based on RTT gradient
- Congestion window management
- Integration with MLT protocol
"""

import time
import math
from typing import Optional, Tuple


class TimelyController:
    """
    TIMELY congestion control algorithm implementation.
    
    Uses RTT measurements to detect congestion and adapt transmission rates.
    The algorithm implements additive increase, multiplicative decrease (AIMD)
    behavior based on RTT gradient and threshold comparisons.
    """
    
    def __init__(
        self,
        initial_rate_mbps: float = 10.0,
        target_rtt_ms: float = 5.0,
        max_rate_mbps: float = 100.0,
        min_rate_mbps: float = 1.0,
        alpha: float = 0.875,  # RTT smoothing factor
        beta: float = 0.8,     # Rate decrease factor
        additive_increase_mbps: float = 10.0,
        rtt_threshold_factor: float = 1.5,  # Multiple of target RTT for threshold
    ):
        """
        Initialize TIMELY controller.
        
        Args:
            initial_rate_mbps: Starting transmission rate in Mbps
            target_rtt_ms: Target RTT in milliseconds
            max_rate_mbps: Maximum allowed transmission rate
            min_rate_mbps: Minimum allowed transmission rate
            alpha: Exponential smoothing factor for RTT (0-1)
            beta: Multiplicative decrease factor (0-1)
            additive_increase_mbps: Additive increase step size
            rtt_threshold_factor: Multiplier for target RTT to set threshold
        """
        self.initial_rate_mbps = initial_rate_mbps
        self.target_rtt_ms = target_rtt_ms
        self.max_rate_mbps = max_rate_mbps
        self.min_rate_mbps = min_rate_mbps
        self.alpha = alpha
        self.beta = beta
        self.additive_increase_mbps = additive_increase_mbps
        self.rtt_threshold_ms = target_rtt_ms * rtt_threshold_factor
        
        # State variables
        self.current_rate_mbps = initial_rate_mbps
        self.smoothed_rtt_ms: Optional[float] = None
        self.rtt_gradient = 0.0
        self.last_rtt_ms: Optional[float] = None
        self.congestion_state = "SLOW_START"  # SLOW_START, CONGESTION_AVOIDANCE, FAST_RECOVERY
        
        # RTT measurement tracking
        self.rtt_samples = []
        self.max_rtt_samples = 10
        
        # Timing for rate calculations
        self.last_update_time = time.time()
        self.bytes_per_second = self._mbps_to_bytes_per_second(self.current_rate_mbps)
        
    def _mbps_to_bytes_per_second(self, mbps: float) -> float:
        """Convert Mbps to bytes per second."""
        return mbps * 1024 * 1024 / 8
    
    def _bytes_per_second_to_mbps(self, bps: float) -> float:
        """Convert bytes per second to Mbps."""
        return bps * 8 / (1024 * 1024)
    
    def start_measurement(self) -> float:
        """
        Start RTT measurement by returning current timestamp.
        
        Returns:
            Current timestamp for RTT calculation
        """
        return time.time()
    
    def update_rtt(self, start_timestamp: float) -> float:
        """
        Update RTT measurement and adjust transmission rate.
        
        Args:
            start_timestamp: Timestamp from start_measurement()
            
        Returns:
            Current transmission rate in bytes per second
        """
        current_time = time.time()
        rtt_ms = (current_time - start_timestamp) * 1000
        
        # Update smoothed RTT using exponential moving average
        if self.smoothed_rtt_ms is None:
            self.smoothed_rtt_ms = rtt_ms
        else:
            self.smoothed_rtt_ms = self.alpha * self.smoothed_rtt_ms + (1 - self.alpha) * rtt_ms
        
        # Calculate RTT gradient (change in RTT)
        if self.last_rtt_ms is not None:
            self.rtt_gradient = rtt_ms - self.last_rtt_ms
        self.last_rtt_ms = rtt_ms
        
        # Store RTT sample for analysis
        self.rtt_samples.append(rtt_ms)
        if len(self.rtt_samples) > self.max_rtt_samples:
            self.rtt_samples.pop(0)
        
        # Apply TIMELY rate control algorithm
        self._apply_rate_control()
        
        # Update bytes per second based on new rate
        self.bytes_per_second = self._mbps_to_bytes_per_second(self.current_rate_mbps)
        self.last_update_time = current_time
        
        return self.bytes_per_second
    
    def _apply_rate_control(self):
        """Apply TIMELY rate control based on RTT measurements."""
        if self.smoothed_rtt_ms is None:
            return
        
        # TIMELY algorithm logic
        if self.smoothed_rtt_ms < self.target_rtt_ms:
            # RTT is below target - increase rate (additive increase)
            self.current_rate_mbps = min(
                self.max_rate_mbps,
                self.current_rate_mbps + self.additive_increase_mbps
            )
            self.congestion_state = "SLOW_START"
            
        elif self.smoothed_rtt_ms > self.rtt_threshold_ms:
            # RTT is significantly above target - decrease rate aggressively
            self.current_rate_mbps = max(
                self.min_rate_mbps,
                self.current_rate_mbps * self.beta
            )
            self.congestion_state = "FAST_RECOVERY"
            
        elif self.rtt_gradient > 0:
            # RTT is increasing (congestion detected) - multiplicative decrease
            decrease_factor = min(self.beta, 1.0 - (self.rtt_gradient / self.target_rtt_ms) * 0.1)
            self.current_rate_mbps = max(
                self.min_rate_mbps,
                self.current_rate_mbps * decrease_factor
            )
            self.congestion_state = "CONGESTION_AVOIDANCE"
            
        else:
            # RTT is stable or decreasing slightly - small additive increase
            self.current_rate_mbps = min(
                self.max_rate_mbps,
                self.current_rate_mbps + self.additive_increase_mbps * 0.1
            )
            self.congestion_state = "CONGESTION_AVOIDANCE"
    
    def get_send_delay(self, chunk_size: int) -> float:
        """
        Calculate delay between sending chunks based on current rate.
        
        Args:
            chunk_size: Size of chunk to be sent in bytes
            
        Returns:
            Delay in seconds before sending next chunk
        """
        if self.bytes_per_second <= 0:
            return 0.0
        
        # Calculate time needed to send this chunk at current rate
        send_time = chunk_size / self.bytes_per_second
        return send_time
    
    def get_current_rate_mbps(self) -> float:
        """Get current transmission rate in Mbps."""
        return self.current_rate_mbps
    
    def get_current_rtt_ms(self) -> Optional[float]:
        """Get current smoothed RTT in milliseconds."""
        return self.smoothed_rtt_ms
    
    def get_stats(self) -> dict:
        """
        Get current statistics for monitoring.
        
        Returns:
            Dictionary with current state and statistics
        """
        return {
            "rate_mbps": self.current_rate_mbps,
            "smoothed_rtt_ms": self.smoothed_rtt_ms,
            "rtt_gradient": self.rtt_gradient,
            "target_rtt_ms": self.target_rtt_ms,
            "congestion_state": self.congestion_state,
            "recent_rtt_samples": self.rtt_samples[-5:] if self.rtt_samples else [],
            "bytes_per_second": self.bytes_per_second,
        }
    
    def reset(self):
        """Reset controller to initial state."""
        self.current_rate_mbps = self.initial_rate_mbps
        self.smoothed_rtt_ms = None
        self.rtt_gradient = 0.0
        self.last_rtt_ms = None
        self.congestion_state = "SLOW_START"
        self.rtt_samples = []
        self.last_update_time = time.time()
        self.bytes_per_second = self._mbps_to_bytes_per_second(self.current_rate_mbps)