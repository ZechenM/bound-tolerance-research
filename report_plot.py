import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend
import matplotlib.pyplot as plt

def plot_test_accuracies(test_accuracies):
    """
    Plots test accuracy curves for multiple sets of accuracy values.
    
    Args:
        test_accuracies (list of list of float): Each inner list contains test accuracy values per epoch.
    """
    plt.figure(figsize=(8, 5))

    # shade BEGIN MID FINAL region to red, yellow, and green
    max_epochs = max(len(acc) for acc, des in test_accuracies)
    begin, mid = 3, 6
    plt.axvspan(0, begin, color='red', alpha=0.3)
    plt.axvspan(begin, mid, color='yellow', alpha=0.3)
    plt.axvspan(mid, max_epochs, color='green', alpha=0.3)

    # plot all lines
    for i, (acc, des) in enumerate(test_accuracies):
        epochs = list(range(0, len(acc)))
        plt.plot(epochs, acc, marker='o', linestyle='-', label=des)
    
    plt.xlabel('Epoch Number')
    plt.ylabel('Test Accuracy')
    plt.title('DenseNet-169 On CIFAR-10 Test Accuracy over Epochs')
    # plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
test_accuracies = [
    # ([0.0985, 0.5138, 0.6405, 0.6966, 0.7306, 0.7858, 0.8013, 0.7912, 0.8164, 0.8268, 0.8337, 0.8287, 0.8373, 0.839, 0.8425],  "3000 per class no drop"),
    ([0.093, 0.4417, 0.5636, 0.6439, 0.6504, 0.727, 0.7573, 0.7343, 0.7832, 0.8006],  "No Drop"),
    ([0.0943, 0.4374, 0.5648, 0.6396, 0.6763, 0.7141, 0.7326, 0.768, 0.7795, 0.7928], "Mid & Final 40% drop"),
    ([0.0943, 0.4176, 0.5326, 0.6576, 0.6848, 0.7185, 0.6996, 0.7555, 0.7555, 0.7665],  "Mid & Final 80% drop"),
    ([0.0943, 0.3442, 0.4201, 0.5731, 0.6331, 0.7113, 0.7198, 0.7523, 0.7855, 0.7978], "Early 40% drop"),
    ([0.0943, 0.2104, 0.3057, 0.4686, 0.5507, 0.6159, 0.6432, 0.6747, 0.7082, 0.7192], "Early 80% drop"),
    # =================================
    # ([0.0985, 0.2953, 0.4496, 0.5293, 0.5602, 0.5997, 0.6214, 0.669, 0.6992, 0.7098, 0.7241, 0.7438, 0.7548, 0.7676, 0.7703], "all 80%"),
    # ([0.0985, 0.1779, 0.1939, 0.1928, 0.2039, 0.2354, 0.1973, 0.2157, 0.2688, 0.2357, 0.2466, 0.2528, 0.2765, 0.2942, 0.2716, 0.2679, 0.2826, 0.3034, 0.2247, 0.3261, 0.3166, 0.3539, 0.2876, 0.3169, 0.339, 0.3597, 0.3601, 0.3811, 0.3412, 0.3764, 0.3787, 0.3601, 0.3636, 0.4123, 0.4295, 0.4438, 0.4394, 0.4368, 0.4237, 0.4556, 0.4446], "all 99%"),

#     Epoch 0: Avg Test Accuracy=0.093, individual=[0.093]
# Epoch 1.0: Avg Test Accuracy=0.4417, individual=[0.4427, 0.4424, 0.4399]
# Epoch 2.0: Avg Test Accuracy=0.5636, individual=[0.5616, 0.5738, 0.5553]
# Epoch 3.0: Avg Test Accuracy=0.6439, individual=[0.6526, 0.6352, 0.644]
# Epoch 4.0: Avg Test Accuracy=0.6504, individual=[0.6459, 0.6552, 0.6501]
# Epoch 5.0: Avg Test Accuracy=0.727, individual=[0.7285, 0.7288, 0.7238]
# Epoch 6.0: Avg Test Accuracy=0.7573, individual=[0.7524, 0.7564, 0.7631]
# Epoch 6.0: Avg Test Accuracy=0.7573, individual=[0.7524, 0.7564, 0.7631]
# Epoch 7.0: Avg Test Accuracy=0.7343, individual=[0.7434, 0.7178, 0.7418]
# Epoch 8.0: Avg Test Accuracy=0.7832, individual=[0.7806, 0.7836, 0.7853]
# Epoch 9.0: Avg Test Accuracy=0.8006, individual=[0.7959, 0.8009, 0.805]
    ([], ""),
    
]
plot_test_accuracies(test_accuracies)
