# bound-tolerance-research

## CLR:

**CLR Detection Parameters in server_compressed.py:**

- **CLR_lst**: List storing CLR detection history as tuples `(batch_number, is_CLR_detected, gradient_norm)`

  - Example: `[(10, True, 1.0), (20, False, 0.5), (30, True, 1.5)]`
  - Tracks when CLR behavior is detected and the corresponding gradient norms
- **CLR_eta**: Threshold for CLR detection (default: 0.5, Accordion author didn't change it)

  - If `|current_norm - previous_norm| / previous_norm >= CLR_eta`, CLR is detected
- **CLR_iter_count**: Counter tracking total number of batches processed from beginning
- - Increments every time `recv_send()` is called
  - Used to determine when to perform CLR detection based on frequency
- **CLR_freq**: Detection frequency in batches (default: 10)

  - CLR detection only runs every `CLR_freq` batches to reduce computational overhead
  - e.g., CLR_freq=10 means detection happens at batches 10, 20, 30, etc.
- **is_CLR_batch**: Boolean flag indicating if current batch exhibits CLR behavior

  - Default: True (first batch is considered CLR by default)
  - Updated after each CLR detection
- **CLR_prev_grad_norm** & **CLR_curr_grad_norm**: Gradient L2 norms for comparison

  - Fast calculation using: `torch.norm(torch.cat([grad.flatten() for grad in gradients.values()]))`

**Implementation Details:**

- From [Accordion](https://github.com/uw-mad-dash/Accordion/) source code, the author detects CLR for EACH LAYER of the model, meaning that one layer could be on CLR but the next layer is not.
- Though the author mentioned that "While batch size scheduling operates at the whole model so ACCORDION looks at the gradient of **whole model** and chooses a suitable batch size.", I fail to see how the author do whole model L2 norm, so here are options:
  1. Flatten all weights, bias, every possible parameter of the model, then do L2 Norm, and see the change ✅ **IMPLEMENTED**
  2. We surely can do per layer Norm comparison, but then how are we going to use this information?
