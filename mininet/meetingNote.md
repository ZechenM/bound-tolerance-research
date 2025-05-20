### Questions

#### What is the mainstream transport protocol that enterprise-level machine learning tasks use? DCTCP? --- This decides what should we test? Test QUBIC? Or DCTCP? Or something else? Just the vanilla tcp reno?

#### When I test effect of drop rate against throughtput on a 100MB setup of 

with each equally drop rate from 0.1% to 2%, I did observe severe performance degradation! 

I used 1M congestion window, MTU 1500. Each rate will be tested for 15 sec. 

server --- switch --- worker1

    |

    |--- worker2

    |

    |--- worker3


#### Linear Scale Plot

![Throughput vs Loss Rate (Linear Scale)](results_20250520_140640/throughput_vs_loss.png)

#### Logarithmic Scale Plot

![Throughput vs Loss Rate (Log Scale)](results_20250520_140640/throughput_vs_loss_log.png)
