import time

class TimelyRateController:
    """
    A standalone implementation of the TIMELY rate control algorithm.
    This class encapsulates the state and logic as described in the SIGCOMM '15 paper.
    """
    def __init__(self, 
                 initial_rate_mbps=10.0, 
                 min_rtt_us=3e4, 
                 tlow_us=5e4, 
                 thigh_us=3e6,
                 alpha=0.05, 
                 beta=0.05, 
                 delta_mbps=4.0):
        
        # Configuration Parameters
        self.min_rtt = min_rtt_us  # Baseline RTT in microseconds
        self.tlow = tlow_us        # Low RTT threshold in microseconds
        self.thigh = thigh_us      # High RTT threshold in microseconds
        self.alpha = alpha         # EWMA weight for RTT difference
        self.beta = beta           # Multiplicative decrease factor
        self.delta = delta_mbps    # Additive increase step in Mbps

        # State Variables
        self.rate_mbps = initial_rate_mbps
        self.prev_rtt = None
        self.rtt_diff = 0.0  # Smoothed RTT difference (the gradient proxy)
        
        # for HAI mode
        self.hai_count = 0

    def on_ack_received(self, new_rtt_us):
        """
        Updates the sending rate based on a new RTT measurement.
        This method implements Algorithm 1 from the TIMELY paper.
        """
        if self.prev_rtt is None:
            self.prev_rtt = new_rtt_us
            return

        # --- Rate Adjustment Logic ---
        reason = ""
        
        # 1. High Latency Safeguard
        if new_rtt_us > self.thigh:
            self.hai_count = 0
            # Aggressive multiplicative decrease
            factor = 1.0 - self.beta * (1.0 - self.thigh / new_rtt_us)
            self.rate_mbps *= factor
            reason = f"Thigh exceeded ({new_rtt_us} > {self.thigh} us)"
        
        # 2. Low Latency Filter
        elif new_rtt_us < self.tlow:
            self.hai_count = 0
            # Additive increase
            self.rate_mbps += self.delta
            reason = f"Below Tlow ({new_rtt_us} < {self.tlow} us)"
        
        # 3. Gradient-Based Control
        else:
            # Calculate and smooth the RTT difference (gradient)
            new_rtt_diff = new_rtt_us - self.prev_rtt
            self.rtt_diff = (1 - self.alpha) * self.rtt_diff + self.alpha * new_rtt_diff
            
            normalized_gradient = self.rtt_diff / self.min_rtt
            
            if normalized_gradient > 0:
                self.hai_count = 0
                # Multiplicative decrease proportional to the gradient
                factor = 1.0 - self.beta * normalized_gradient
                # TODO: necessary? Clamp factor to avoid excessive reduction
                self.rate_mbps *= max(0.5, factor) 
                reason = f"Positive Gradient ({normalized_gradient:.2f})"
            else:
                # Additive increase
                self.hai_count += 1
                self.rate_mbps += self.hai_count * self.delta
                reason = f"Negative/Zero Gradient ({normalized_gradient:.2f})"

        # Ensure rate does not fall below a minimum threshold
        self.rate_mbps = max(self.rate_mbps, self.delta)
        
        self.prev_rtt = new_rtt_us
        
        return reason

    def get_sending_rate_mbps(self):
        return self.rate_mbps
