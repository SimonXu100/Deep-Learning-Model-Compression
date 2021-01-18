import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plot_performance(file_pth, save_file_pth, title, num_params_file=None):
    results = np.loadtxt(file_pth)
    if num_params_file:
        num_params = np.loadtxt(num_params_file)
        x = num_params[3]/num_params[1]
        x = np.concatenate([[1], x])*100
    else:
        x = results[0]

    # Use inverted x axis to better illustrate the decrease in performance as
    # fraction of parameters left decrease
    plt.gca().invert_xaxis()
    # Since each time we are pruning x% of parameters, x axis should be in log scale
    plt.xscale("log")
    # Format x axis ticks as percentage
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    # Display tick at x values of datapoints
    plt.xticks(x)
    plt.minorticks_off()
    plt.xlabel("Percentage of parameters left")
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.plot(x, results[1], label="Test Accu before finetune")
    plt.plot(x, results[2], label="Train Accu before finetune")
    plt.plot(x, results[3], label="Test Accu after finetune")
    plt.plot(x, results[4], label="Train Accu after finetune")

    plt.legend()
    plt.savefig(save_file_pth)
    plt.show()


def compare_l1_random():
    l1_result = np.loadtxt("performance/Alexnet_unstructured_l1_performance.txt")
    random_result =  np.loadtxt("performance/Alexnet_unstructured_random_performance.txt")
    x = l1_result[0]

    plt.figure(figsize=(8,4))
    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.xticks(x)
    plt.minorticks_off()
    plt.xlabel("Percentage of weights kept")
    plt.ylabel("Accuracy")
    plt.title("AlexNet Unstructured L1 vs Random")

    offset = 0.25*x
    plt.bar(x + offset/2, l1_result[2], color='b', width=0.25*(x + offset/2), label='L1')
    plt.bar(x - offset/2, random_result[2], color='r', width=0.25*(x - offset/2), label='Random')

    plt.legend()
    plt.savefig("plots/Alexnet_l1_vs_random.jpg")
    plt.show()


def compare_num_params():
    num_params = np.loadtxt("performance/Alexnet_structured_num_params.txt")
    x = np.arange(len(num_params[0]))

    plt.figure(figsize=(12, 6))
    # Set the label of each bar
    plt.xticks(x, list(map("{:.0f}%".format, num_params[0])))
    plt.xlabel("Percentage of weights kept according to Pytorch")
    plt.ylabel("Number of parameters")
    plt.title("AlexNet Pytorch vs Actual Number of Parameters")

    # Offset x so the the tick is between two bars. Use 0.33/2 since the
    # position of bar is defined as the center of the bar horizontally
    plt.bar(x-0.33/2, num_params[2], color='b', width=0.33, label='Num params reported by Pytorch')
    plt.bar(x+0.33/2, num_params[3], color='r', width=0.33, label='Actual Num Params')

    plt.legend()
    plt.savefig("plots/Alexnet_num_params.jpg")
    plt.show()


def plot_efficiency(file_pth, save_file_pth, title):
    results = np.loadtxt(file_pth)
    x = results[0]

    plt.xlabel("Batch Size")
    plt.ylabel("Inference time (ms)")
    plt.title(title)

    plt.plot(x, results[1], marker='o', label="Original")
    plt.plot(x, results[2], marker='o', label="Pytorch Pruned")
    plt.plot(x, results[3], marker='o', label="Actually pruned")

    plt.legend()
    plt.savefig(save_file_pth)
    plt.show()


def compare_efficiency(file_pths, num_params_file, save_file_pth, title, batch_size):
    results = []
    for file_pth in file_pths:
        results.append(np.loadtxt(file_pth))
    results = np.array(results)

    num_params = np.loadtxt(num_params_file)
    x = num_params[3] / num_params[1]
    x = np.concatenate([[1], x]) * 100

    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.xticks(x,  list(map("{:.0f}%".format, x)))
    plt.minorticks_off()
    plt.xlabel("Percentage of parameters")
    plt.ylabel("Inference time (ms)")
    plt.title(title)

    # Find the column in the file with the desired batch size
    # Assuming all efficiency file comparing here has the same set of batch sizes
    batch_size_index = np.where(results[0, 0, :] == batch_size)[0][0]
    print("Current Batch Size:", results[0, 0, batch_size_index])
    y = np.concatenate([[np.mean(results[:, 1, batch_size_index])], results[:, 3, batch_size_index]])
    print(y)
    plt.plot(x, y)

    plt.savefig(save_file_pth)
    plt.show()


def compare_prune_frac_BERT():
    results_16 = np.loadtxt("performance/BERT_unstructured_0.16_performance.txt")
    results_30 = np.loadtxt("performance/BERT_unstructured_0.3_performance.txt")
    results_50 = np.loadtxt("performance/BERT_unstructured_0.5_performance.txt")

    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.xticks(results_16[0], list(map("{:.0f}%".format, results_16[0])))
    plt.minorticks_off()
    plt.xlabel("Percentage of parameters left")
    plt.ylabel("Accuracy")
    plt.title("Test Accu of BERT with Different Pruning Fractions")

    plt.plot(results_16[0], results_16[3], label="Prune Frac = 0.1591")
    plt.plot(results_30[0], results_30[3], label="Prune Frac = 0.2929")
    plt.plot(results_50[0, 0:4], results_50[3, 0:4], label="Prune Frac = 0.5")

    plt.legend()
    plt.savefig("plots/BERT_compare_prune.jpg")
    plt.show()


if __name__ == "__main__":
    plot_performance("performance/Alexnet_unstructured_l1_performance.txt",
                     "plots/Alexnet_unstructured_l1_performance.jpg",
                     "AlexNet Unstructured L1")
    compare_l1_random()

    plot_performance("performance/Alexnet_structured_conv_0.1_performance.txt",
                     "plots/Alexnet_structured_performance.jpg",
                     "AlexNet Structured Pruning",
                     num_params_file="performance/Alexnet_structured_num_params.txt")
    compare_num_params()
    frac = "53.14"
    plot_efficiency("performance/Alexnet_structured_%s_efficiency.txt"%frac,
                    "plots/Alexnet_structured_%s_efficiency.jpg"%frac,
                    "Inference time of pruned model on RTX 2070")
    file_pths = []
    frac = 90
    for i in range(10):
        file_pths.append("performance/Alexnet_structured_%.2f_efficiency.txt"%frac)
        frac *= 0.9
    compare_efficiency(file_pths, "performance/Alexnet_structured_num_params.txt",
                       "plots/Alexnet_structured_compare_efficiency.jpg",
                       "Inference time on RTX 2070 with Batch Size 16", 16)

    plot_performance("performance/BERT_unstructured_0.16_performance.txt",
                     "plots/BERT_unstructured_0.16_performance.jpg",
                     "BERT Unstructured Frac=0.1591")
    compare_prune_frac_BERT()
