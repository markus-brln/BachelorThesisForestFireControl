import matplotlib.pyplot as plt
import numpy as np

def barplot_with_error(data, labels, ylabel):
    if len(data) != len(labels):
        print("data and labels must have the same length!")
        return

    means = []
    SDs = []
    for array in data:
        array = np.array(array)
        means.append(np.mean(array))
        SDs.append(np.std(array))
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x_pos, means,
           yerr=SDs,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    #plt.savefig('bar_plot_with_error_bars.png')
    plt.show()


def parse_file(filename):
    print("parsing", filename)
    import ast
    file = open(filename,mode='r')

    all_burned = []
    n_successes = []

    lines = file.readlines()
    for idx, line in enumerate(lines[1:]):
        if idx % 2 == 0:
            n_successes.append(len(ast.literal_eval(line)))
            all_burned.extend(ast.literal_eval(line))
            print(len(ast.literal_eval(line)), sep=" ")

    return n_successes, all_burned


if __name__ == "__main__":
    files = ["xy/xySTOCHASTIC.txt","xy/xyUNCERTAINONLY.txt","xy/xyWINDONLY.txt","xy/xyUNCERTAIN+WIND.txt"]

    success_data = []
    burned_data = []
    for file in files:
        n_successes, all_burned = parse_file(file)
        success_data.append(n_successes)
        burned_data.append(all_burned)

    labels = ["STOCHASTIC", "UNCERTAIN", "WIND", "UNCERTAIN+WIND"]
    barplot_with_error(success_data, labels, "Mean successfully contained fires out of 100")
    barplot_with_error(burned_data, labels, "Mean amount of burned cells")