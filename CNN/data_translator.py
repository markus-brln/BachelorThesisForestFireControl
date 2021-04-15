import os
import numpy as np
import glob
import matplotlib.pyplot as plt
sep = os.path.sep

def load_all_data():
  """Loads and concatenates all data from files in data/runs/"""
  dirname = os.path.dirname(os.path.realpath(__file__)) + sep + ".." + sep + "gui" + sep + "data" + sep

  filepaths = glob.glob(dirname + "runs" + sep +  "*.npy")
  if not filepaths:
    print("no files found at: ", dirname + "runs" + sep)
    exit()

  data = np.load(filepaths[0], allow_pickle=True)
  for filepath in filepaths[1:]:                      # optionally load more data
    file_data = np.load(filepath, allow_pickle=True)
    data = np.concatenate(data, file_data)

  return data


def raw_to_IO_arrays(data):
  """The raw data consists of data points which each consist of the 2D environment image,
     the wind direction and the wind speed. It is transformed to """

  # INPUT AND OUTPUT PICTURES
  n_channels = np.max(data[0][0])           # the highest number gives the amount of different pixels
                                            # minus waypoint pixels -> channels
  waypoint_channel = 5
  shape = (len(data), np.shape(data[0][0])[0], np.shape(data[0][0])[1], n_channels)
  images = np.zeros(shape, dtype=np.uint8)
  shape = (len(data), np.shape(data[0][0])[0], np.shape(data[0][0])[1])
  waypoint_imgs = np.zeros(shape, dtype=np.uint8)


  for i in range(len(data)):
    picture_raw = data[i][0]
    for y, row in enumerate(picture_raw):
      for x, cell in enumerate(row):
        if cell == waypoint_channel:        # make 2D image of waypoints
          waypoint_imgs[i][y][x] = 1
          images[i][y][x][0] = 1            # leave waypoints like forest
        else:
          images[i][y][x][cell] = 1


  # INPUT WIND SPEED AND WIND DIRECTION VECTORS
  shape = (len(data), len(data[0][1]) + len(data[0][2]))
  print(shape)
  wind_dir_speed = np.zeros(shape, dtype=np.uint8)

  for i in range(len(data)):                # make one array 13x1 for concatenation
    wind_dir_speed[i] = np.concatenate([data[i][1], data[i][2]])

  # BUILD INPUT AND OUTPUT ARRAYS
  input = list()                            # standard input img + wind info to be concatenated
  output = waypoint_imgs                    # just to be specific about what is output
  for i in range(len(data)):
    input.append([images[i], wind_dir_speed[i]])

  input = np.asarray(input)

  print("input len: ", len(input))
  print("output len: ", len(output))
  print("image shape: ", np.shape(input[0][0]))
  print("wind info vector shape: ", np.shape(input[0][1]))

  return input, output



if __name__ == "__main__":
  data = load_all_data()
  inputs, outputs = raw_to_IO_arrays(data)

  inputs, outputs = np.un
