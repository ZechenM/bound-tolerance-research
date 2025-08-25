# bound-tolerance-research

## CLR (Critical Learning Regime):

**CLR Detection Parameters in server_multithreading.py:**

- **CLR_lst**: List storing CLR detection history as tuples `(CLR_iter_count, is_CLR_iter, gradient_norm)`

  - Example: `[(10, True, 1.0), (20, False, 0.5), (30, True, 1.5)]`
  - Tracks when CLR behavior is detected and the corresponding gradient norms
- **CLR_eta**: Threshold for CLR detection (default: 0.5, Accordion author didn't change it)

  - If `|current_norm - previous_norm| / previous_norm >= CLR_eta`, CLR is detected
- **CLR_iter_count**: Counter tracking total number of server-worker communication rounds from beginning

  - Increments every time gradient aggregation completes in `gradient_aggregator_loop()`
  - Used to determine when to perform CLR detection based on frequency
- **CLR_freq**: Detection frequency in communication rounds (default: 10)

  - CLR detection only runs every `CLR_freq` communication rounds to reduce computational overhead
  - e.g., CLR_freq=10 means detection happens at communication rounds 10, 20, 30, etc.
- **is_CLR_iter**: Boolean flag indicating if current communication round exhibits CLR behavior

  - Default: True (first communication round is considered CLR by default)
  - Updated after each CLR detection
- **CLR_prev_grad_norm** & **CLR_curr_grad_norm**: Gradient L2 norms for comparison

  - Fast calculation using: `torch.norm(torch.cat([grad.flatten() for grad in gradients.values()]))`

**Configuration Parameters (config.py):**

- **USE_CLR**: Enable/disable CLR detection (default: False)

  - When False: No CLR detection is performed, saving computation overhead
  - When True: Full CLR detection and loss_tolerance control is enabled
- **CLR_eta**: Threshold for CLR detection (default: 0.5)
- **CLR_freq**: Detection frequency in communication rounds (default: 10)

  - Also determines the duration of loss_tolerance=0 period after CLR detection

**Implementation Details:**

- **Detection Timing**: CLR detection is performed every `CLR_freq` communication rounds between server and workers, NOT based on epochs

  - This approach is more suitable for distributed training where workers may have different training speeds
  - Detection frequency can be adjusted by modifying `CLR_freq` parameter
- **Gradient Source**: CLR detection uses averaged gradients from all workers after aggregation

  - Ensures detection is based on the collective learning behavior across all workers
  - More robust than single-worker gradient analysis
- **Integration**: CLR detection is integrated into the `gradient_aggregator_loop()` method

  - Runs after gradient averaging is complete
  - Minimal impact on training performance
- **Time Tracking**: All CLR events are timestamped with millisecond precision

  - CLR detection events include precise timestamps (YYYY-MM-DD HH:MM:SS.mmm format)
  - Loss tolerance changes are also timestamped for complete audit trail
  - Enables precise analysis of CLR timing and duration

**Loss-Tolerance Control:**

- **Dynamic Adjustment**: When CLR is detected, `current_loss_tolerance` is automatically set to 0
- **Duration Control**: Loss tolerance remains at 0 for `CLR_freq` communication rounds (same as detection frequency)
- **Automatic Restoration**: After the CLR period ends, loss tolerance is restored to the base `loss_tolerance` value
- **Real-time Updates**: MLT protocol uses `get_current_loss_tolerance()` to get the current value
- **Network Impact**: During CLR periods, the system waits for all data packets (no early stopping)
- **Consistency**: The duration of loss_tolerance=0 period automatically matches the CLR detection frequency

**Key Differences from Original Accordion Implementation:**

- From [Accordion](https://github.com/uw-mad-dash/Accordion/) source code, the author detects CLR for EACH LAYER of the model, meaning that one layer could be on CLR but the next layer is not.
- Though the author mentioned that "While batch size scheduling operates at the whole model so ACCORDION looks at the gradient of **whole model** and chooses a suitable batch size.", I fail to see how the author do whole model L2 norm, so here are options:
  1. Flatten all weights, bias, every possible parameter of the model, then do L2 Norm, and see the change ✅ **IMPLEMENTED**

**Future Work:**

- **Epoch Integration**: Currently epoch information is set to 0 as a placeholder
  - TODO: Integrate with actual epoch tracking from worker data
  - Note: Current machines only train for < 5 epochs, so epoch-based detection may not be necessary
- **Loss-Tolerance Strategy**: CLR detection results can be used to dynamically adjust MLT tolerance values
  - Higher loss tolerance during non-CLR periods
  - Lower loss tolerance during CLR periods to preserve learning quality
