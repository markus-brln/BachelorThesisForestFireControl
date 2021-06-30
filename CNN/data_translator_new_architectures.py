#!/bin/python3

import os
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from numpy.lib.function_base import angle
from NNutils import plot_np_image, plot_data
import sys
import time

sep = os.path.sep
timeframe = 20
size = 255
apd = 10                                                    # agent_point_diameter of inactive agents in CNN input



def load_raw_data(file_filter):
  """Loads and concatenates all data from files in data/runs/"""
  dirname = os.path.dirname(os.path.realpath(__file__)) + sep + ".." + sep + "gui" + sep + "data" + sep

  filepaths = glob.glob(dirname + "runs" + sep +  "*.npy")

  if not filepaths:
    print("no files found at: ", dirname + "runs" + sep)
    exit()

    ## Filter data by filename
  fp_tmp = list()
  for filepath in filepaths:
    if file_filter in filepath:
      fp_tmp.append(filepath)
  filepaths = fp_tmp

  print("loading data from:", filepaths, sep='\n')
  data = np.load(filepaths[0], allow_pickle=True)
  for filepath in filepaths[1:]:                      # optionally load more data
    if file_filter in filepath:
      print(filepath)
      file_data = np.load(filepath, allow_pickle=True)
      data = np.concatenate([data, file_data])
  return data


def outputs_xy(data):
  print("Constructing xy outputs")
  agent_info = [data_point[3] for data_point in data]
  agent_info = [j for sub in agent_info for j in sub]       # flatten the list

  outputs = []                                              # [[x rel. to agent, y, drive/dig], ...]
  for raw in agent_info:
    agent_pos = raw[0]                                      # make things explicit, easy-to-understand
    wp = raw[1]
    drive_dig = raw[2]

    max_dist = timeframe
    if drive_dig == 0:                                      # driving wp => 2 times the speed
      max_dist = 2 * timeframe

    delta_x = (wp[0] - agent_pos[0]) / max_dist             # normalized difference between agent position and wp
    delta_y = (wp[1] - agent_pos[1]) / max_dist

    outputs.append([delta_x, delta_y, drive_dig])

  #print(outputs)
  #print(agent_info)
  return np.asarray(outputs, dtype=np.float16)


def outputs_segments(data):
  print("Constructing angle segments outputs")
  agent_info = [data_point[3] for data_point in data]
  agent_info = [j for sub in agent_info for j in sub]  # flatten the list

  outputs = []  # [[x rel. to agent, y, drive/dig], ...]
  for raw in agent_info:

    agent_pos = raw[0]  # make things explicit, easy-to-understand
    wp = raw[1]
    dig_drive = raw[2]

    delta_x = wp[0] - agent_pos[0]
    delta_y = wp[1] - agent_pos[1]

    angle = math.atan2(delta_y, delta_x)
    segments = [0] * 16

    segments[round(8 * angle / (2 * math.pi)) % 16] = 1

    print(segments)
    outputs.append(segments + [dig_drive])

  print(outputs[:10])
  return np.asarray(outputs, dtype=np.float16)


def outputs_angle(data):
  print("Constructing angle outputs")
  agent_info = [data_point[3] for data_point in data]
  agent_info = [j for sub in agent_info for j in sub]  # flatten the list

  outputs = []  # [[x rel. to agent, y, drive/dig], ...]
  for raw in agent_info:

    agent_pos = raw[0]  # make things explicit, easy-to-understand
    wp = raw[1]
    drive_dig = raw[2]

    max_dist = timeframe
    if drive_dig == 0:  # driving wp => 2 times the speed
      max_dist = 2 * timeframe

    delta_x = (wp[0] - agent_pos[0]) / max_dist  # normalized difference between agent position and wp
    delta_y = (wp[1] - agent_pos[1]) / max_dist

    angle = math.atan2(delta_y, delta_x)
    cos_position = math.cos(angle)
    sin_position = math.sin(angle)

    radius = math.sqrt(delta_x ** 2 + delta_y ** 2)
    outputs.append([cos_position, sin_position, radius, drive_dig])

  # print(outputs)
#   print(agent_info)
  return np.asarray(outputs, dtype=np.float16)


def waypoint2array(wp):
  boxsize = 5
  arr = {}
  x = 0
  cur_entry = 0
  to_ret = -1
  for y in range(-boxsize, boxsize + 1):
    for entry in range(-x, x + 1):
      arr[cur_entry] = (entry, y)
      cur_entry += 1
    x += 1 if y < 0 else -1
  for idx in range(len(arr)):
    if (arr[idx] == wp):
      to_ret = idx
  return to_ret

def shrink2reachablewaypoint(wpX, wpY):
  '''
    fits waypoints into a range that the agent can reach given timeframe
  '''
  size = 4
  timeframe = 20
  boxsize = timeframe / size
  roundError = 0.0
  diff = timeframe - (abs(wpX) + abs(wpY))
  if diff > size:
    multi = timeframe/diff
    tf = multi * boxsize
    boxsize = tf/boxsize
  while(float(abs(wpX) + abs(wpY)) > boxsize):
    scale = boxsize / (abs(wpX) + abs(wpY))
    wpX = (wpX * scale) - roundError
    wpY = (wpY * scale) - roundError
    wpX = round(wpX)
    wpY = round(wpY)
    roundError += 0.01
    if(roundError >= 0.5):
      # print("round", roundError)
      break
  wp = (wpX, wpY)
  return tuple(wp)

def outputs_box(data):
  '''
  returns a list of two parts the first a 1D vector of size timeframe^2 + (timeframe+1)^2
    with a 1 for the position of the correct waypoint location in format [0,0,0,1,...,0]
    secondly a value of 0 for drive and 1 for dig for the type of waypoint for each agent
  '''
# TODO make this work
  print("Constructing box output")
  agent_info = [data_point[3] for data_point in data] ## list of agent location, waypoint and dig/drive
  agent_info = [j for sub in agent_info for j in sub]  # flatten the list to be unique per agent
  outputs = []
  boxsize = 5
  for agent in agent_info:
    # box/grid of possible locations format [[0,0,1,0,..], dig/drive],[[0,1,...0], dig/drive...]
    output = [0] * ((boxsize * boxsize) + ((boxsize + 1) * (boxsize + 1)))
    # print("len", len(output))
    xpos, ypos = agent[0]
    waypoint = tuple(agent[1])
    # print("agent: (", xpos, ",", ypos, ") wp: (", waypoint[0], ",", waypoint[1], ")")
    drive_dig = agent[2]
    deltaX = int((waypoint[0] - xpos) + 1)
    deltaY = int((waypoint[1] - ypos) + 1)
    # print("(", deltaX, deltaY, ")")
    wp = shrink2reachablewaypoint(deltaX, deltaY)
    arrPos = waypoint2array(wp)
    if arrPos != -1:
      output[arrPos] = 1
    else:
      print("Scale fail!")
    list = output + [drive_dig]
    # print(list)
    outputs.append(list)
  return np.asarray(outputs, dtype=np.float16)


def construct_output(data, NN_variant):
  """
    3 different outputs based on architecture variant:
  - x, y (normalized)
  - angle, distance
  - vector of L*L where L == side length of a box around agent, 1 where agent needs to go
  """
  output = []
  if NN_variant == "xy":
    output = outputs_xy(data)
  if NN_variant == "angle":
    output = outputs_angle(data)
  if NN_variant == "box":
    output = outputs_box(data)
  if NN_variant == "segments":
    output = outputs_segments(data)

  return output

def construct_input(data):
  """
  Translates the raw data generated from playing the simulation to NN-friendly I/O.
  7 channel image NN input:
  - [0] active fire (no burned cell or tree channels)
  - [1] fire breaks
  - [2] wind direction
  - [3] wind speed
  - [4] other agents
  - [5] active agent x
  - [6] active agent y
  """
  # DEFINITIONS
  n_channels = 7
  n_agents = len(data[0][3])

  # FIX N_AGENTS NOT SAME (DEAD AGENT IN DATA)
  datatmp = []
  for data_point in data:
    agent_specific = data_point[3]
    if len(agent_specific) == n_agents:  # some data points don't have the right amount of agents
      datatmp.append(data_point)
  data = datatmp#[0:5]
  print(len(data))

  # INPUT IMAGES
  shape = (len(data), 256, 256, n_channels)  # new pass per agent
  input_images_single = np.zeros(shape, dtype=np.float16)  # single images not for each agent

  for i in range(len(data)):
    if i % 100 == 0:
      print("picture " + str(i) + "/" + str(len(data)))
    picture_raw = data[i][0]

    for y, row in enumerate(picture_raw):  # picture of environment (trees and burned cells ignored)
      for x, cell in enumerate(row):
        if cell == 1:  # firebreak channel [1]
          input_images_single[i][y][x][1] = 1
        if cell == 2:  # active fire channel [0]
          input_images_single[i][y][x][0] = 1

    wind_dir_idx = np.argmax(data[i][1])  # encode wind dir and speed info in channels [2] and [3]
    wind_speed_idx = np.argmax(data[i][2])
    input_images_single[i][:, :, 2] = wind_dir_idx / (len(data[i][1]) - 1)
    input_images_single[i][:, :, 3] = wind_speed_idx / (len(data[i][2]) - 1)

  apd = 10  # agent_point_diameter
  all_images = []  # multiply amount of images by n_agents, set channels [4] and [5]
  active_agents_pos = []
  for single_image, data_point, i in zip(input_images_single, data, range(0, len(data))):
    if i % 100 == 0:
      print("picture per agent " + str(i) + "/" + str(len(data)))
    agent_positions = [agent[0] for agent in data_point[3]]

    for active_pos in agent_positions:
      agent_image = np.copy(single_image)
      # agent_image[active_pos[0] - apd : active_pos[0] + apd, active_pos[1] - apd : active_pos[1] + apd, 5] = 1      # mark position of active agent, channel [5]
      active_agents_pos.append([active_pos[0] / 255, active_pos[1] / 255])

      for others_pos in agent_positions:
        if others_pos != active_pos:
          # agent_image[others_pos[0]][others_pos[1]][4] = 1  # mark position of other agents, channel [4]
          agent_image[others_pos[1] - apd: others_pos[1] + apd, others_pos[0] - apd: others_pos[0] + apd,
          4] = 1

      # plot_np_image(agent_image)
      all_images.append(agent_image)  # 1 picture per agent

  # FOR NON-CONCAT
  i = 0
  for img, agent in zip(all_images, active_agents_pos):
    img[:, :, 5] = agent[0]  # x, y position of active agent on channel 5,6
    img[:, :, 6] = agent[1]
    # if (i + 5) % 5 == 0:
    #   print("episode", int(i / 5))
    # print("agent:", round(agent[0] * 255), round(agent[1] * 255))
    i += 1
    # plot_np_image(img)

  #print(active_agents_pos)
  print("final amount of datapoints: ", len(all_images))

  return np.asarray(all_images, dtype=np.float16)  # , np.asarray(active_agents_pos)


def raw_to_IO(data, NN_variant):
  outputs = construct_output(data, NN_variant)
  images = construct_input(data)  # same input data for each architecture

  return images, outputs

def augmentData(data):
  print("!augmenting data!")
  print("original", data[-1])
  augmentedData = data.tolist()
  print(len(augmentedData))
  for data_point in data:
    agent_positions = data_point[3]
    for i in range(1, 3):
      ''' x - 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0] - i, agent[0][1])
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), aug_positions]
      augmentedData.append(episode)
      ''' x - 1, y - 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0] - i, agent[0][1] - i)
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), aug_positions]
      augmentedData.append(episode)
      ''' y - 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0], agent[0][1] - i)
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), aug_positions]
      augmentedData.append(episode)
      ''' x + 1, y - 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0] + i, agent[0][1] - i)
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), aug_positions]
      augmentedData.append(episode)
      ''' x + 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0] + i, agent[0][1])
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), aug_positions]
      augmentedData.append(episode)
      ''' x + 1, y + 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0] + i, agent[0][1] + i)
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), aug_positions]
      augmentedData.append(episode)
      ''' y + 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0], agent[0][1] + i)
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), list(aug_positions)]
      augmentedData.append(episode)
      ''' x - 1, y + 1 '''
      aug_positions = []
      for agent in agent_positions:
        temp = agent.copy()
        temp[0] = (agent[0][0] - i, agent[0][1] + i)
        aug_positions.append(temp)
        episode = [np.asarray(data_point[0]), np.asarray(data_point[1]), np.asarray(data_point[2]), list(aug_positions)]
      augmentedData.append(episode)
  return  np.asarray(augmentedData, dtype=object)

if __name__ == "__main__":
  print(os.path.realpath(__file__))
  architecture_variants = ["xy", "angle", "box", "segments"]             # our 3 individual network output variants
  experiments = ["BASIC", "STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
  if len(sys.argv) > 1 and int(sys.argv[1]) < len(architecture_variants):
    out_variant = architecture_variants[int(sys.argv[1])]

  else:
    out_variant = architecture_variants[3]
  if len(sys.argv) > 2 and int(sys.argv[2]) < len(experiments):
      experiment = experiments[int(sys.argv[4])]
  else:
      experiment = experiments[0]
  data = load_raw_data(file_filter=experiment)
  data = data[:]
  if out_variant == 'box':
    data = augmentData(data)
  # data = shift_augment(data)     # does not work yet
  # print("lenny", len(data))
  #plot_data(data)

  print(out_variant)
  images, outputs = raw_to_IO(data, out_variant)

  NN_number = 0
  np.save(file="images_" + out_variant + experiment + ".npy", arr=images, allow_pickle=True)   # save to here, so the CNN dir
  #np.save(file="concat_" + out_variant + ".npy", arr=concat, allow_pickle=True)
  np.save(file="outputs_" + out_variant + experiment + ".npy", arr=outputs, allow_pickle=True)
