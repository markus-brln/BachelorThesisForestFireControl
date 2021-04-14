import os
import numpy as np
import glob
import matplotlib.pyplot as plt
sep = os.path.sep

def play():
  dirname = os.path.dirname(os.path.realpath(__file__)) + sep + ".." + sep + "gui" + sep + "data" + sep
  filepath = dirname + "runs" + sep + "run0.npy"

  array = np.load(filepath, allow_pickle=True)

  for i in range(len(array)):
    plt.imshow(array[i][0])
    plt.show()

  #for file in glob.glob(dirname + "runs" + sep + "*.npy"):
    #filenames.append(file)


if __name__ == "__main__":


  play()