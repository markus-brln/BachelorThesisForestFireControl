import os
import random

import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from numpy.core import multiarray
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
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
  print(filepaths[0])

  for filepath in filepaths[1:]:                      # optionally load more data
    if file_filter in filepath:
      print(filepath)
      file_data = np.load(filepath, allow_pickle=True)
      data = np.concatenate([data, file_data])


  return data

def rot_pos(x, y):
  size = 256
  x -= size / 2
  y -= size / 2

  return (-y + size / 2, x + size / 2)


def rotate(datapoint):
  to_return = np.array(datapoint.shape)
  new_waypoints = []
  for waypoint in datapoint[4]:
    new_waypoints += [[rot_pos(waypoint[0])], rot_pos(waypoint[1]), waypoint[2]]


def augmentations(data):
  newData = []
  for idx in range(data):
    data_to_add = [data]
    data_to_add.append(rotate(data_to_add[-1]))
    data_to_add.append(rotate(data_to_add[-1]))
    data_to_add.append(rotate(data_to_add[-1]))
    newData += data_to_add

  return data



def raw_to_IO_arrays(data):
  """See Documentation/dataTranslationNewArchitecture.png"""
  # DEFINITIONS
  n_channels = 5
  n_agents = 5
  env_dim = 256
  waypoint_dig_channel = 5
  waypoint_drive_channel = 6

  # FIX N_AGENTS NOT SAME
  datatmp = []
  for data_point in data:
    agent_specific = data_point[3]
    if len(agent_specific) == n_agents:       # some data points don't have the right amount of agents
      datatmp.append(data_point)
  data = datatmp
  print(len(data))

  shape = (len(data), 256, 256, n_channels)      # new pass per agent
  images_single = np.zeros(shape, dtype=np.uint8)           # single images not for each agent
  shape = (len(data), 256, 256, 3)
  waypoint_imgs = np.zeros(shape, dtype=np.uint8)

  print(len(data))


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
    #if len(agent_specific) == n_agents:       # some data points don't have the right amount of agents
    for j in range(n_agents):
      agent_positions.append([agent_specific[j][0][0] / env_dim, agent_specific[j][0][1] / env_dim])             # x, y of agent that needs waypoint
      #print(j, [agent_specific[j][0][0] / env_dim, agent_specific[j][0][1] / env_dim])
      outputs.append([agent_specific[j][1][0] / env_dim, agent_specific[j][1][1] / env_dim, agent_specific[j][2]])
    #else:
    #  images = np.delete(images, range(i, i+5), 0)
    #  wind_dir_speed = np.delete(wind_dir_speed, range(i, i+5), 0)


  agent_positions = np.asarray(agent_positions, dtype=np.float)
  outputs = np.asarray(outputs, dtype=np.float)

  # CONCAT WIND + AGENT POS
  concat_vector = list()
  for wind, agentpos in zip(wind_dir_speed, agent_positions):
    concat_vector.append(list(wind) + list(agentpos))
  concat_vector = np.asarray(concat_vector)


  print("input image shape: ", np.shape(images))
  print("wind info vector shape: ", np.shape(wind_dir_speed))
  print("agent input positions: ", np.shape(agent_positions))
  print("wind+agent concat: ", concat_vector.shape)
  print("outputs shape: ", np.shape(outputs))





  # PLOT TO CHECK RESULTS
  #for i, dat in enumerate(data):
  #  print("wind info: ", wind_dir_speed[5 * i : 5 * i + 4])
  #  print("agent pos: ", agent_positions[5 * i : 5 * i + 4])
  #  print("output waypoints: ", outputs[5 * i : 5 * i + 4])
  #  img = dat[0]
  #  plt.imshow(np.reshape(img, (255, 255)))
  #  plt.show()

        # input, input,         output
  return images, concat_vector, outputs



if __name__ == "__main__":
  print(os.path.realpath(__file__))

  data = load_all_data(file_filter="NEWFive")
  print(len(data))
  print(type(data))
  print(data[0])
  exit(0)
  images, concat, outputs = raw_to_IO_arrays(data)

  np.save(file="imagesNEW.npy", arr=images, allow_pickle=True)   # save to here, so the CNN dir
  np.save(file="concatNEW.npy", arr=concat, allow_pickle=True)
  np.save(file="outputsNEW.npy", arr=outputs, allow_pickle=True)
