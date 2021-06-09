import os
import random

import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from numpy.core import multiarray
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
sep = os.path.sep
timeframe = 20
size = 255

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

  data = np.load(filepaths[0], allow_pickle=True)
  for filepath in filepaths[1:]:                      # optionally load more data
    if file_filter in filepath:
      print(filepath)
      file_data = np.load(filepath, allow_pickle=True)
      data = np.concatenate([data, file_data])
  return data


def rot_pos(pos):
  size = 255
  x, y = pos
  x -= size / 2
  y -= size / 2

  return (-y + size / 2, x + size / 2)


def rotate_wind(wind):
    list = wind.tolist()
    idx = list.index(1)
    list[idx] = 0
    if idx == 0: #nn
      idx = 2 #ee
    elif idx == 1: #ss
      idx = 3 #ww
    if idx == 2: #ee
      idx = 1 #ss
    elif idx == 3: #ww
      idx = 0 #nn
    if idx == 4: #ne
      idx = 6 #se
    elif idx == 5: #nw
      idx = 4 #ne
    if idx == 6: #se
      idx = 7 #sw
    elif idx == 7: #sw
      idx = 5 #nw
    list[idx] = 1
    wind = np.array(list)
    return wind


def rotate(datapoint):
  environment = np.rot90(datapoint[0])
  wind = rotate_wind(datapoint[1])
                                    # Wind speed
  windspeed = datapoint[2]

  new_waypoints = []
  for idx in range(len(datapoint[3])):
    first_entry = rot_pos(datapoint[3][idx][0])
    second_entry = rot_pos(datapoint[3][idx][1])
    digging = datapoint[3][idx][2]
    new_waypoints.append([first_entry, second_entry, digging])
    
  return [environment, wind, windspeed, new_waypoints]


def augment_datapoint(datapoint):
  augmented_data = [datapoint]
  augmented_data.append(rotate(augmented_data[-1]))
  augmented_data.append(rotate(augmented_data[-1]))
  augmented_data.append(rotate(augmented_data[-1]))

  return augmented_data


def raw_to_IO_arrays(data):
  """See Documentation/dataTranslationNewArchitecture.png"""
  # DEFINITIONS
  n_channels = 5
  n_agents = 4
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
  # print(len(data))

  shape = (len(data), 256, 256, n_channels)      # new pass per agent
  images_single = np.zeros(shape, dtype=np.uint8)           # single images not for each agent
  shape = (len(data), 256, 256, 3)
  waypoint_imgs = np.zeros(shape, dtype=np.uint8)

  # print(len(data))


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
      agent_positions.append([agent_specific[j][0][0] / env_dim, agent_specific[j][0][1] / env_dim])     # x, y of agent that needs waypoint
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


def plot_np_image(image):
  channels = np.dsplit(image.astype(dtype=np.float32), len(image[0][0]))
  f, axarr = plt.subplots(2, 4)
  axarr[0, 0].imshow(np.reshape(channels[0], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[0, 0].set_title("active fire")
  axarr[0, 1].imshow(np.reshape(channels[1], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[0, 1].set_title("fire breaks")
  axarr[0, 2].imshow(np.reshape(channels[2], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[0, 2].set_title("wind dir (uniform)")
  axarr[1, 0].imshow(np.reshape(channels[3], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 0].set_title("wind speed (uniform)")
  axarr[1, 1].imshow(np.reshape(channels[4], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 1].set_title("other agents")
  axarr[1, 2].imshow(np.reshape(channels[5], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 2].set_title("active agent x")
  axarr[1, 3].imshow(np.reshape(channels[6], newshape=(256, 256)), vmin=0, vmax=1)
  axarr[1, 3].set_title("active agent y")
  print("max", np.max(channels[5]), np.max(channels[6]))
  plt.show()


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

  print(outputs)
  print(agent_info)
  return np.asarray(outputs, dtype=np.float16)


def outputs_angle(data):
  print("Constructing xy outputs")
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

    angle = math.atan2(delta_y, delta_x),

    outputs.append([angle, drive_dig])

  print(outputs)
  print(agent_info)
  return np.asarray(outputs, dtype=np.float16)


def outputs_box(data):
# TODO make this work
  print("Constructing box output")
  agent_info = [data_point[3] for data_point in data] ## list of agent location, waypoint and dig/drive
  agent_info = [j for sub in agent_info for j in sub]  # flatten the list to be unique per agent

  outputs = []  # box/grid of possible locations

  for agent in agent_info:
    xpos, ypos = agent[0]
    for x in range(1, timeframe + 1, 1):
      newXpos = xpos + x
      newYpos = ypos + timeframe - x
      if (newXpos >= 0 and newXpos <= size):
        if (newYpos >= 0 and newYpos <= size):
          a = (newXpos, newYpos)
          outputs.append(a)
      if (x == timeframe):
        newXpos = xpos + timeframe - x
        newYpos = ypos + x
        if (newXpos >= 0 and newXpos <= size):
          if (newYpos >= 0 and newYpos <= size):
            b = (newXpos, newYpos)
      else:
        newXpos = xpos + timeframe - x
        newYpos = ypos - x
        if (newXpos >= 0 and newXpos <= size):
          if (newYpos >= 0 and newYpos <= size):
            b = (newXpos, newYpos)
      if b:
        outputs.append(b)
      newXpos = xpos - x
      newYpos = ypos + timeframe - x
      if (newXpos >= 0 and newXpos <= size):
        if (newYpos >= 0 and newYpos <= size):
          c = (newXpos, newYpos)
          outputs.append(c)

      newXpos = xpos - timeframe + x
      newYpos = ypos - x
      if (newXpos >= 0 and newXpos <= size):
        if (newYpos >= 0 and newYpos <= size):
          d = (newXpos, newYpos)
          outputs.append(d)

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

  return output


def construct_input(data):
  """
  Translates the raw data generated from playing the simulation to NN-friendly I/O.

  6 channel image NN input: (5 channels atm)
  - [0] active fire (no burned cell or tree channels)
  - [1] fire breaks
  - [2] wind direction
  - [3] wind speed
  - [4] other agents
  - NOT USED [5] current agent (maybe mark several pixels)
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
    print("picture " + str(i) + "/" + str(len(data)))
    picture_raw = data[i][0]

    for y, row in enumerate(picture_raw):                   # picture of environment (trees and burned cells ignored)
      for x, cell in enumerate(row):
        if cell == 1:                                       # firebreak channel [1]
          input_images_single[i][y][x][1] = 1
        if cell == 2:                                       # active fire channel [0]
          input_images_single[i][y][x][0] = 1

    wind_dir_idx = np.argmax(data[i][1])                    # encode wind dir and speed info in channels [2] and [3]
    wind_speed_idx = np.argmax(data[i][2])
    input_images_single[i][:, :, 2] = wind_dir_idx / (len(data[i][1]) - 1)
    input_images_single[i][:, :, 3] = wind_speed_idx / (len(data[i][2]) - 1)


  apd = 10                                                   # agent_point_diameter
  all_images = []                                           # multiply amount of images by n_agents, set channels [4] and [5]
  active_agents_pos = []
  for single_image, data_point, i in zip(input_images_single, data, range(0, len(data))):
    print("picture per agent " + str(i) + "/" + str(len(data)))
    agent_positions = [agent[0] for agent in data_point[3]]

    for active_pos in agent_positions:
      agent_image = np.copy(single_image)
      #agent_image[active_pos[0] - apd : active_pos[0] + apd, active_pos[1] - apd : active_pos[1] + apd, 5] = 1      # mark position of active agent, channel [5]
      active_agents_pos.append([active_pos[0] / 255, active_pos[1] / 255])

      for others_pos in agent_positions:
        if others_pos != active_pos:
          #agent_image[others_pos[0]][others_pos[1]][4] = 1  # mark position of other agents, channel [4]
          agent_image[others_pos[1] - apd: others_pos[1] + apd, others_pos[0] - apd : others_pos[0] + apd, 4] = 1

      #plot_np_image(agent_image)
      all_images.append(agent_image)                        # 1 picture per agent

  # FOR NON-CONCAT
  for img, agent in zip(all_images, active_agents_pos):
    img[:, :, 5] = agent[0]                                 # x, y position of active agent on channel 5,6
    img[:, :, 6] = agent[1]
    #plot_np_image(img)

  print(active_agents_pos)
  print("final amount of datapoints: ",len(all_images))

  return np.asarray(all_images, dtype=np.float16)#, np.asarray(active_agents_pos)


def construct_input_images_only(data):
  """
  Translates the raw data generated from playing the simulation to NN-friendly I/O.

  6 channel image NN input:
  - [0] active fire (no burned cell or tree channels)
  - [1] fire breaks
  - [2] wind direction
  - [3] wind speed
  - [4] other agents
  - [5] current agent (maybe mark several pixels)
  """
  # DEFINITIONS
  n_channels = 6
  n_agents = len(data[0][3])

  # FIX N_AGENTS NOT SAME (DEAD AGENT IN DATA)
  datatmp = []
  for data_point in data:
    agent_specific = data_point[3]
    if len(agent_specific) == n_agents:  # some data points don't have the right amount of agents
      datatmp.append(data_point)
  data = datatmp#[:5]
  print(len(data))

  # INPUT IMAGES
  shape = (len(data), 256, 256, n_channels)  # new pass per agent
  input_images_single = np.zeros(shape, dtype=np.float16)  # single images not for each agent

  for i in range(len(data)):
    print("picture " + str(i) + "/" + str(len(data)))
    picture_raw = data[i][0]

    for y, row in enumerate(picture_raw):                   # picture of environment (trees and burned cells ignored)
      for x, cell in enumerate(row):
        if cell == 1:                                       # firebreak channel [1]
          input_images_single[i][y][x][1] = 1
        if cell == 2:                                       # active fire channel [0]
          input_images_single[i][y][x][0] = 1

    wind_dir_idx = np.argmax(data[i][1])                    # encode wind dir and speed info in channels [2] and [3]
    wind_speed_idx = np.argmax(data[i][2])
    input_images_single[i][:, :, 2] = wind_dir_idx / (len(data[i][1]) - 1)
    input_images_single[i][:, :, 3] = wind_speed_idx / (len(data[i][2]) - 1)

  apd = 10                                                   # agent_point_diameter
  all_images = []                                           # multiply amount of images by n_agents, set channels [4] and [5]
  for single_image, data_point, i in zip(input_images_single, data, range(0, len(data))):
    print("picture per agent " + str(i) + "/" + str(len(data)))
    agent_positions = [agent[0] for agent in data_point[3]]

    for active_pos in agent_positions:
      agent_image = np.copy(single_image)
      agent_image[active_pos[0] - apd : active_pos[0] + apd, active_pos[1] - apd : active_pos[1] + apd, 5] = 1      # mark position of active agent, channel [5]

      for others_pos in agent_positions:
        if others_pos != active_pos:
          #agent_image[others_pos[0]][others_pos[1]][4] = 1  # mark position of other agents, channel [4]
          agent_image[others_pos[0] - apd: others_pos[0] + apd, others_pos[1] - apd : others_pos[1] + apd, 4] = 1

      #plot_np_image(agent_image)
      all_images.append(agent_image)                        # 1 picture per agent

  print("final amount of datapoints: ",len(all_images))

  return np.asarray(all_images, dtype=np.float16)



def construct_input_bigAgents(data):  ## not used afaik
  """
  Translates the raw data generated from playing the simulation to NN-friendly I/O.

  6 channel image NN input: (5 channels atm)
  - [0] active fire (no burned cell or tree channels)
  - [1] fire breaks
  - [2] wind direction
  - [3] wind speed
  - [4] other agents
  - NOT USED [5] current agent (maybe mark several pixels)
  """
  # DEFINITIONS
  n_channels = 5
  n_agents = len(data[0][3])

  # FIX N_AGENTS NOT SAME (DEAD AGENT IN DATA)
  datatmp = []
  for data_point in data:
    agent_specific = data_point[3]
    if len(agent_specific) == n_agents:  # some data points don't have the right amount of agents
      datatmp.append(data_point)
  data = datatmp#[:5]
  print(len(data))

  # INPUT IMAGES
  shape = (len(data), 256, 256, n_channels)  # new pass per agent
  input_images_single = np.zeros(shape, dtype=np.float16)  # single images not for each agent

  for i in range(len(data)):
    print("picture " + str(i) + "/" + str(len(data)))
    picture_raw = data[i][0]

    for y, row in enumerate(picture_raw):                   # picture of environment (trees and burned cells ignored)
      for x, cell in enumerate(row):
        if cell == 1:                                       # firebreak channel [1]
          input_images_single[i][y][x][1] = 1
        if cell == 2:                                       # active fire channel [0]
          input_images_single[i][y][x][0] = 1

    wind_dir_idx = np.argmax(data[i][1])                    # encode wind dir and speed info in channels [2] and [3]
    wind_speed_idx = np.argmax(data[i][2])
    input_images_single[i][:, :, 2] = wind_dir_idx / (len(data[i][1]) - 1)
    input_images_single[i][:, :, 3] = wind_speed_idx / (len(data[i][2]) - 1)

  apd = 10                                                   # agent_point_diameter
  all_images = []                                           # multiply amount of images by n_agents, set channels [4] and [5]
  active_agents_pos = []
  for single_image, data_point, i in zip(input_images_single, data, range(0, len(data))):
    print("picture per agent " + str(i) + "/" + str(len(data)))
    # agent_positions = [agent[0] for agent in data_point[3]]
    for agent in data_point[3]:
      agent_x, agent_y = agent[0]
      neighbouringCells = []
      neighbouringCells.append([agent_x - 1, agent_y - 1])

    for active_pos in agent_positions:
      agent_image = np.copy(single_image)
      #agent_image[active_pos[0] - apd : active_pos[0] + apd, active_pos[1] - apd : active_pos[1] + apd, 5] = 1      # mark position of active agent, channel [5]
      active_agents_pos.append([active_pos[0] / 255, active_pos[1] / 255])

      for others_pos in agent_positions:
        if others_pos != active_pos:
          #agent_image[others_pos[0]][others_pos[1]][4] = 1  # mark position of other agents, channel [4]
          agent_image[others_pos[0] - apd: others_pos[0] + apd, others_pos[1] - apd : others_pos[1] + apd, 4] = 1

      #plot_np_image(agent_image)
      all_images.append(agent_image)                        # 1 picture per agent

  print(active_agents_pos)
  print("final amount of datapoints: ",len(all_images))

  return np.asarray(all_images, dtype=np.float16), np.asarray(active_agents_pos)



def raw_to_IO(data, NN_variant):
  outputs = construct_output(data, NN_variant)
  if NN_variant == 'input':
    images, concat = construct_input_bigAgents(data) ## creates a box around the agent to increase weight of agent position in the model
  else:
    images = construct_input(data)  # same input data for each architecture

  return images, outputs

def plot_data(data):
  for dat in data:
    print(dat[3])
    plt.imshow(dat[0])
    plt.show()

  exit()

if __name__ == "__main__":
  print(os.path.realpath(__file__))
  data = load_raw_data(file_filter="mXYEASYFOUR1")#"mXYEASYFIVE")
  data = data[:100]

  #plot_data(data)

  architecture_variants = ["xy", "angle", "box"]             # our 3 individual network output variants
  out_variant = architecture_variants[0]
  images, outputs = raw_to_IO(data, out_variant)

  #for img, out in zip(images, outputs):
  #  print(out)
  #  print("x, y: ", img[0][0][5], img[0][0][6])
  #  plot_np_image(img)

  #exit()

  np.save(file="images_" + out_variant + ".npy", arr=images, allow_pickle=True)   # save to here, so the CNN dir
  #np.save(file="concat_" + out_variant + ".npy", arr=concat, allow_pickle=True)
  np.save(file="outputs_" + out_variant + ".npy", arr=outputs, allow_pickle=True)
