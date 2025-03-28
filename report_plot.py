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
    begin, mid = 3, 5
    plt.axvspan(0, begin, color="red", alpha=0.3)
    plt.axvspan(begin, mid, color="yellow", alpha=0.3)
    plt.axvspan(mid, max_epochs, color="green", alpha=0.3)

    # plot all lines
    for i, (acc, des) in enumerate(test_accuracies):
        epochs = list(range(1, len(acc) + 1))
        plt.plot(epochs, acc, marker="o", linestyle="-", label=des)

    plt.xlabel("Epoch Number")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy over Epochs")
    # plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
begin_test_accuracies = [
    (
        [0.0985, 0.5138, 0.6405, 0.6966, 0.7306, 0.7858, 0.8013, 0.7912],
        "3000 per class no drop",
    ),
    (
        [0.0985, 0.4561, 0.5921, 0.6579, 0.7102, 0.7672, 0.7866, 0.8016],
        "only begin 40% drop",
    ),
    (
        [0.0985, 0.3430, 0.4342, 0.5768, 0.6699, 0.7201, 0.7533, 0.7787],
        "only begin 80% drop",
    ),
    # ([0.0985, 0.4352, 0.5753, 0.6546, 0.7072, 0.7576, 0.78, 0.7992], "begin 40%, mid 0, final 40%"),
    # ([0.0985, 0.2953, 0.4496, 0.5293, 0.5602, 0.5997, 0.6214, 0.669, 0.6992, 0.7098, 0.7241, 0.7438, 0.7548, 0.7676, 0.7703], "all 80%"),
    # ([0.0985, 0.1779, 0.1939, 0.1928, 0.2039, 0.2354, 0.1973, 0.2157, 0.2688, 0.2357, 0.2466, 0.2528, 0.2765, 0.2942, 0.2716, 0.2679, 0.2826, 0.3034, 0.2247, 0.3261, 0.3166, 0.3539, 0.2876, 0.3169, 0.339, 0.3597, 0.3601, 0.3811, 0.3412, 0.3764, 0.3787, 0.3601, 0.3636, 0.4123, 0.4295, 0.4438, 0.4394, 0.4368, 0.4237, 0.4556, 0.4446], "all 99%"),
    ([], ""),
]

mid_test_accuracies = [
    (
        [0.0985, 0.5138, 0.6405, 0.6966, 0.7306, 0.7858, 0.8013, 0.7912],
        "3000 per class no drop",
    ),
    (
        [0.0985, 0.5262, 0.6333, 0.7087, 0.7492, 0.7888, 0.7979, 0.8155],
        "only mid 40% drop",
    ),
    (
        [0.0985, 0.5303, 0.6449, 0.7128, 0.7423, 0.7703, 0.7938, 0.8117],
        "only mid 80% drop",
    ),
]


final_test_accuracies = [
    (
        [0.0985, 0.5138, 0.6405, 0.6966, 0.7306, 0.7858, 0.8013, 0.7912],
        "3000 per class no drop",
    ),
    (
        [0.0985, 0.5249, 0.6370, 0.7103, 0.7530, 0.7746, 0.8009, 0.8163],
        "only final 40% drop",
    ),
    (
        [0.0985, 0.5448, 0.6371, 0.7257, 0.7538, 0.7826, 0.7862, 0.8082],
        "only final 80% drop",
    ),
]

test_accuracies = final_test_accuracies
plot_test_accuracies(test_accuracies)
