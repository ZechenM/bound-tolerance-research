import matplotlib

matplotlib.use("TkAgg")  # Use Tkinter backend
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
    begin, mid = 6, 7
    plt.axvspan(0, begin, color="red", alpha=0.3)
    plt.axvspan(begin, mid, color="yellow", alpha=0.3)
    plt.axvspan(mid, max_epochs, color="green", alpha=0.3)

    # plot all lines
    for i, (acc, des) in enumerate(test_accuracies):
        epochs = list(range(1, len(acc) + 1))
        plt.plot(epochs, acc, marker="o", linestyle="-", label=des)

    plt.xlabel("Epoch Number")
    plt.ylabel("Test Accuracy")
    plt.title("DenseNet Test Accuracy over Epochs")
    # plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
test_accuracies = [
    (
        [0.0950, 0.3043, 0.5065, 0.5778, 0.6085, 0.6457, 0.6708, 0.6926, 0.7119, 0.7383],
        "1200 per class no drop",
    ),
    (
        [0.0950, 0.2511, 0.3488, 0.3325, 0.4078, 0.4549, 0.5356, 0.5442, 0.5925, 0.6325],
        "only begin 80% drop",
    ),
    (
        [0.0950, 0.3270, 0.4590, 0.5412, 0.5861, 0.6260, 0.6555, 0.7067, 0.7069, 0.7403],
        "vt begin 80% drop",
    ),
    # ([0.0985, 0.4352, 0.5753, 0.6546, 0.7072, 0.7576, 0.78, 0.7992], "begin 40%, mid 0, final 40%"),
    # ([0.0985, 0.2953, 0.4496, 0.5293, 0.5602, 0.5997, 0.6214, 0.669, 0.6992, 0.7098, 0.7241, 0.7438, 0.7548, 0.7676, 0.7703], "all 80%"),
    # ([0.0985, 0.1779, 0.1939, 0.1928, 0.2039, 0.2354, 0.1973, 0.2157, 0.2688, 0.2357, 0.2466, 0.2528, 0.2765, 0.2942, 0.2716, 0.2679, 0.2826, 0.3034, 0.2247, 0.3261, 0.3166, 0.3539, 0.2876, 0.3169, 0.339, 0.3597, 0.3601, 0.3811, 0.3412, 0.3764, 0.3787, 0.3601, 0.3636, 0.4123, 0.4295, 0.4438, 0.4394, 0.4368, 0.4237, 0.4556, 0.4446], "all 99%"),
    ([], ""),
]
plot_test_accuracies(test_accuracies)
