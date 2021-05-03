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

  fp_tmp = list()
  for filepath in filepaths:
    if file_filter in filepath:
      fp_tmp.append(filepath)

  filepaths = fp_tmp

  data = np.load(filepaths[0], allow_pickle=True)

  for filepath in filepaths[1:]:                      # optionally load more data
    if file_filter in filepath:
      print(filepath)
      file_data = np.load(filepath, allow_pickle=True)
      data = np.concatenate([data, file_data])

  return data


def raw_to_IO_arrays(data):
  """"""
  # DEFINITIONS
  data = data[:2]          # TODO convert everything
  n_channels = 5
  n_agents = 5
  env_dim = 256
  waypoint_dig_channel = 5
  waypoint_drive_channel = 6
  shape = (len(data), 256, 256, n_channels)      # new pass per agent
  images_single = np.zeros(shape, dtype=np.uint8)           # single images not for each agent
  shape = (len(data), 256, 256, 3)
  waypoint_imgs = np.zeros(shape, dtype=np.uint8)

  # INPUT IMAGES
  for i in range(len(data)):
    print("picture " + str(i) + "/" + str(len(data)))
    picture_raw = data[i][0]
    for y, row in enumerate(picture_raw):
      for x, cell in enumerate(row):
        if cell == waypoint_dig_channel:         # make 2D image of waypoints
          images_single[i][y][x][0] = 1             # leave waypoints like forest (so at idx 0 -> value 1)
        elif cell == waypoint_drive_channel:         # make 2D image of waypoints
          images_single[i][y][x][0] = 1             # leave waypoints like forest (so at idx 0 -> value 1)
        else:
          images_single[i][y][x][cell] = 1

  shape = (len(data) * n_agents, 256, 256, n_channels)      # new pass per agent
  images = np.zeros(shape, dtype=np.uint8)           # images for each agent

  for i, img in enumerate(images_single):           # multiply data by n_agents
    for j in range(n_agents):
      images[i * n_agents + j] = img


  # INPUT WIND SPEED AND WIND DIRECTION VECTORS
  shape = (len(data), len(data[0][1]) + len(data[0][2]))
  wind_dir_speed_single = np.zeros(shape, dtype=np.uint8)
  for i in range(len(data)):                 # make one array 13x1 for concatenation
    wind_dir_speed_single[i] = np.concatenate([data[i][1], data[i][2]])

  shape = (len(data) * n_agents, len(data[0][1]) + len(data[0][2]))
  wind_dir_speed = np.zeros(shape, dtype=np.uint8)

  for i, wind in enumerate(wind_dir_speed_single):           # multiply data by n_agents
    for j in range(n_agents):
      wind_dir_speed[i * n_agents + j] = wind

  wind_dir_speed = np.asarray(wind_dir_speed)



  # BUILD OUTPUT COORDINATES + dig/drive ID 1/0
  agent_positions = list()                   # x, y to be concatenated
  outputs = list()                           # x, y, 0/1 drive/dig


  for i in range(len(data)):
    print("output " + str(i) + "/" + str(len(data)))
    agent_specific = data[i][3]
    print(agent_specific)
    for j in range(n_agents):
      agent_positions.append([agent_specific[j][0][0] / env_dim, agent_specific[j][0][1] / env_dim])             # x, y of agent that needs waypoint
      outputs.append([agent_specific[j][1][0] / env_dim, agent_specific[j][1][1] / env_dim, agent_specific[j][2]])


  agent_positions = np.asarray(agent_positions, dtype=np.float)
  outputs = np.asarray(outputs, dtype=np.float)
  print(agent_positions)
  print(outputs)



  print("input image shape: ", np.shape(images))
  print("wind info vector shape: ", np.shape(wind_dir_speed))
  print("agent input positions: ", np.shape(agent_positions))
  print("outputs shape: ", np.shape(outputs))


  # PLOT TO CHECK RESULTS
  for dat in data:
    img = dat[0]
    plt.imshow(np.reshape(img, (255, 255)))
    plt.show()


  return images_single, wind_dir_speed, outputs



if __name__ == "__main__":
  # TODO 3 dimensional (normal, dig, drive), softmax activation, pixelwise softmax
  print(os.path.realpath(__file__))
  data = load_all_data(file_filter="NEWFive")
  images, wind_dir_speed, outputs = raw_to_IO_arrays(data)

  np.save(file="images_old.npy", arr=images, allow_pickle=True)   # save to here, so the CNN dir
  np.save(file="windinfo_old.npy", arr=wind_dir_speed, allow_pickle=True)
  np.save(file="outputs_old.npy", arr=outputs, allow_pickle=True)
