import matplotlib.pyplot as plt
import numpy as np

def barplot_with_error(all_data, labels, ylabel):
    """
    Function making a barplot with arbitrary amounts of bars.

    :param all_data: list of lists of data. Each sublist should contain the same amount of elements.
    :param labels: of len(all_data[0])
    :param ylabel: Text describing the y-axis.
    """
    if len(all_data[0]) != len(labels):
        print("data and labels must have the same length!")
        return

    all_means = []
    all_SDs = []
    for data in all_data:
        means = []
        SDs = []
        for array in data:
            array = np.array(array)
            means.append(np.mean(array))
            SDs.append(np.std(array))
        all_means.append(means)
        all_SDs.append(SDs)

    x_pos = np.arange(len(labels))

    architectures = ['xy', 'angle', 'box', 'segments']
    architecture_colors = ['blue', 'green', 'orange', 'red']

    bar_width = 0.2
    fig, ax = plt.subplots()
    for means, SDs, idx in zip(all_means, all_SDs, range(len(all_data))):
        barlist = ax.bar(x_pos - (((len(all_data) - 1) / 2) - idx) * bar_width,
               all_means[idx],
               yerr=all_SDs[idx],
               align='center',
               alpha=0.8,
               width=bar_width,
               ecolor='black',
               color=architecture_colors[idx],
               capsize=10)


    handles = [plt.Rectangle((0, 0), 1, 1, color=label) for label in architecture_colors]
    ax.legend(handles, architectures)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)

    #plt.tight_layout()
    #plt.savefig('bar_plot_containment_rate.png')
    plt.show()


def parse_file(filename):
    print("parsing", filename)
    import ast
    file = open(filename, mode='r')

    all_burned = []
    n_successes = []

    lines = file.readlines()
    for idx, line in enumerate(lines[1:]):
        if idx % 2 == 0:
            n_successes.append(len(ast.literal_eval(line)))
            all_burned.extend(ast.literal_eval(line))
            #print(len(ast.literal_eval(line)), sep=" ")

    return n_successes, all_burned


if __name__ == "__main__":
    files1 = ["xy/xySTOCHASTIC.txt", "xy/xyUNCERTAINONLY.txt", "xy/xyWINDONLY.txt", "xy/xyUNCERTAIN+WIND.txt"]
    files2 = ["angle/angleSTOCHASTIC.txt", "angle/angleUNCERTAINONLY.txt", "angle/angleWINDONLY.txt", "angle/angleUNCERTAIN+WIND.txt"]
    files3 = ["box/boxSTOCHASTIC.txt", "box/boxUNCERTAINONLY.txt", "box/boxWINDONLY.txt", "box/boxUNCERTAIN+WIND.txt"]
    files4 = ["segments/segmentsSTOCHASTIC.txt", "segments/segmentsUNCERTAINONLY.txt", "segments/segmentsWINDONLY.txt",
              "segments/segmentsUNCERTAIN+WIND.txt"]
    all_success = []
    all_burned = []
    file_lists = [files1, files2, files3, files4]
    for file_list in file_lists:
        success_data = []
        burned_data = []
        for file in file_list:
            n_successes, burned_cells = parse_file(file)
            success_data.append(n_successes)
            burned_data.append(burned_cells)
        all_success.append(success_data)
        all_burned.append(burned_data)

    labels = ["BASELINE", "UNCERTAIN", "WIND", "UNCERTAIN+WIND"]
    barplot_with_error(all_success, labels, "Mean successfully contained fires out of 100")
    barplot_with_error(all_burned, labels, "Mean amount of burned cells")