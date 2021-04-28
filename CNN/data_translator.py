import os
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
sep = os.path.sep

def load_all_data(file_filter):
  """Loads and concatenates all data from files in data/runs/"""
  dirname = os.path.dirname(os.path.realpath(__file__)) + sep + ".." + sep + "gui" + sep + "data" + sep

  filepaths = glob.glob(dirname + "runs" + sep +  "*.npy")
  if not filepaths:
    print("no files found at: ", dirname + "runs" + sep)
    exit()

  print(len(filepaths))

  data = np.load(filepaths[0], allow_pickle=True)
  for filepath in filepaths[1:]:                      # optionally load more data
    if file_filter in filepath:
      print(filepath)                                 # TODO jFive0.npy not there??
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
  n_channels = 5
  waypoint_dig_channel = 5
  waypoint_drive_channel = 6
  shape = (len(data), 256, 256, n_channels)
  images = np.zeros(shape, dtype=np.uint8)
  shape = (len(data), 256, 256, 2)
  waypoint_imgs = np.zeros(shape, dtype=np.uint8)


  for i in range(10):
    print("picture " + str(i) + "/" + str(len(data)))
    picture_raw = data[i][0]
    for y, row in enumerate(picture_raw):
      for x, cell in enumerate(row):
        if cell == waypoint_dig_channel:         # make 2D image of waypoints
          waypoint_imgs[i][y][x][0] = 1
          images[i][y][x][0] = 1             # leave waypoints like forest (so at idx 0 -> value 1)
        elif cell == waypoint_drive_channel:         # make 2D image of waypoints
          waypoint_imgs[i][y][x][1] = 1
          images[i][y][x][0] = 1             # leave waypoints like forest (so at idx 0 -> value 1)
        else:
          images[i][y][x][cell] = 1


  # INPUT WIND SPEED AND WIND DIRECTION VECTORS
  shape = (len(data), len(data[0][1]) + len(data[0][2]))
  wind_dir_speed = np.zeros(shape, dtype=np.uint8)

  for i in range(len(data)):                 # make one array 13x1 for concatenation
    wind_dir_speed[i] = np.concatenate([data[i][1], data[i][2]])

  wind_dir_speed = np.asarray(wind_dir_speed)

  # BUILD OUTPUT IMAGES, 255x255 to 64x64x3 (normal, dig waypoint, drive waypoint) -> pixel wise softmax
  outputs = np.zeros((len(data), 64, 64, 3), dtype=np.uint8)

  for i in range(10):#len(data)):
    print("downscale picture " + str(i) + "/" + str(len(data)))
    for y, row in enumerate(waypoint_imgs[i]):
      for x, cell in enumerate(row):
        outx = math.floor(x / 4)
        outy = math.floor(y / 4)
        if cell[0] > outputs[i][outy][outx][0]:
          outputs[i][outy][outx][1] = 1
        elif cell[1] > outputs[i][outy][outx][1]:
          outputs[i][outy][outx][2] = 1
        else:
          outputs[i][outy][outx][0] = 1


  print("input examples len: ", len(images))
  print("output examples len: ", len(outputs))
  print("image shape: ", np.shape(images[0]))
  print("wind info vector shape: ", np.shape(wind_dir_speed[0]))
  print("output image shape: ", np.shape(outputs[0]))

  # PLOT TO CHECK RESULTS (difficult for 3D input images)
  #for outp, full in zip(outputs, waypoint_imgs):                       # output 2D images
  #  plt.imshow(full)
  #  plt.show()
  #  plt.imshow(np.reshape(outp, newshape=(64,64)))
  #  plt.show()
  #for inp in inputs:                         # wind info vectors (need two 1 values)
  #  plt.plot(inp[1])
  #  plt.show()

  return images, wind_dir_speed, outputs



if __name__ == "__main__":
  # TODO 3 dimensional (normal, dig, drive), softmax activation, pixelwise softmax
  print(os.path.realpath(__file__))
  data = load_all_data(file_filter="Five")
  images, wind_dir_speed, outputs = raw_to_IO_arrays(data)

  np.save(file="images.npy", arr=images, allow_pickle=True)   # save to here, so the CNN dir
  np.save(file="windinfo.npy", arr=wind_dir_speed, allow_pickle=True)
  np.save(file="outputs.npy", arr=outputs, allow_pickle=True)
