import os
import numpy as np
import glob
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
    data = np.concatenate([data, file_data])

  return data


def raw_to_IO_arrays(data):
  """The raw data consists of data points which each consist of the 2D environment image,
     the wind direction and the wind speed. It is transformed to input and output arrays,
     where:
     inputs = len(data) * [3D_image_without_waypoints, wind dir + speed vector]
     outputs = len(data) * 2D_image_of_waypoints
     For details see /Documentation/dataTranslation.png"""

  # INPUT AND OUTPUT PICTURES
  n_channels = np.max(data[0][0])            # the highest number gives the amount of different pixels
                                             # minus waypoint pixels -> channels
  waypoint_channel = 5
  shape = (len(data), np.shape(data[0][0])[0], np.shape(data[0][0])[1], n_channels)
  images = np.zeros(shape, dtype=np.uint8)
  shape = (len(data), np.shape(data[0][0])[0], np.shape(data[0][0])[1])
  waypoint_imgs = np.zeros(shape, dtype=np.uint8)


  for i in range(len(data)):
    print("picture " + str(i) + "/" + str(len(data)))
    picture_raw = data[i][0]
    for y, row in enumerate(picture_raw):
      for x, cell in enumerate(row):
        if cell == waypoint_channel:         # make 2D image of waypoints
          waypoint_imgs[i][y][x] = 1
          images[i][y][x][0] = 1             # leave waypoints like forest (so at idx 0 -> value 1)
        else:
          images[i][y][x][cell] = 1


  # INPUT WIND SPEED AND WIND DIRECTION VECTORS
  shape = (len(data), len(data[0][1]) + len(data[0][2]))
  wind_dir_speed = np.zeros(shape, dtype=np.uint8)

  for i in range(len(data)):                 # make one array 13x1 for concatenation
    wind_dir_speed[i] = np.concatenate([data[i][1], data[i][2]])

  wind_dir_speed = np.asarray(wind_dir_speed)

  # BUILD INPUT AND OUTPUT ARRAYS
  #inputs = list()                            # standard input img + wind info to be concatenated
  outputs = waypoint_imgs                    # just to be specific about what is output
  #for i in range(len(data)):
  #  inputs.append([images[i], wind_dir_speed[i]])
  #inputs = [images, wind_dir_speed]
  #inputs = np.asarray(inputs)

  print("input examples len: ", len(images))
  print("output examples len: ", len(outputs))
  print("image shape: ", np.shape(images[0]))
  print("wind info vector shape: ", np.shape(wind_dir_speed[0]))

  # PLOT TO CHECK RESULTS (difficult for 3D input images)
  #for outp in outputs:                       # output 2D images
  #  plt.imshow(outp)
  #  plt.show()
  #for inp in inputs:                         # wind info vectors (need two 1 values)
  #  plt.plot(inp[1])
  #  plt.show()

  return images, wind_dir_speed, outputs



if __name__ == "__main__":
  # TODO set up CNN, ask how training can be done with 2 inputs and concatenation

  print(os.path.realpath(__file__))
  data = load_all_data()
  images, wind_dir_speed, outputs = raw_to_IO_arrays(data)

  np.save(file="images.npy", arr=images, allow_pickle=True)   # save to here, so the CNN dir
  np.save(file="windinfo.npy", arr=wind_dir_speed, allow_pickle=True)
  np.save(file="outputs.npy", arr=outputs, allow_pickle=True)
