import glob
import platform
import numpy as np
import os
import matplotlib.pyplot as plt

from Model.node import NodeState
from Model.utils import *
from Model.direction import Direction

class DataSaver:
  def __init__(self, model):
    self.model = model
    self.episode_data = list()                              # gathered during episode, not sure yet whether it will succeed
    self.all_data = list()                                  # working with lists, later converted to np array
    self.name = None


  def append_datapoint(self):
    """Should get called when new agent waypoints were set and model is about to
       fast-forward some timesteps (see utils.py, 'timeframe') to see how it played out."""
    for agent in self.model.agents:                         # do not append data points when agents are dead,
      if agent.dead:                                        # would break things down the line
        return

    image = np.zeros((self.model.size, self.model.size), dtype=np.uint8)  # save memory with uint8
    wind_dir_vec = np.zeros(n_wind_dirs, dtype=np.uint8)                  # 8 wind directions
    windspeed_vec = np.zeros(n_wind_speed_levels, dtype=np.uint8)         # 5 'wind speed levels'
    agent_pos_with_waypoints = list()                       # elements: [agent.position,
                                                            #            agent.waypoint_digging,
                                                            #            drive/dig (0/1)]

    #  set NodeState to be pixel value:
    #  NORMAL = 0
    #  FIREBREAK = 1
    #  ON_FIRE = 2
    #  BURNED_OUT = 3
    #  AGENT = 4
    #  waypoint_digging = 5 (based on agent.waypoint)
    #  waypoint_walking = 6
    for x, node_row in enumerate(self.model.nodes):         # make a picture of the node state
      for y, node in enumerate(node_row):
        image[y][x] = node.state                            # NOTE: x,y flipped because array is accessed via y,x == row, col

    for agent in self.model.agents:
      x, y = agent.position
      image[y][x] = int(NodeState.AGENT)
      if agent.waypoint_digging:                            # one of the waypoints is None
        x, y = agent.waypoint_digging
        image[y][x] = 5
        agent_pos_with_waypoints.append([agent.position, agent.waypoint_digging, 1])    # 1 == digging
      if agent.waypoint_walking:
        x, y = agent.waypoint_walking
        image[y][x] = 6
        agent_pos_with_waypoints.append([agent.position, agent.waypoint_walking, 0])    # 0 == driving

    wind_dir_vec[self.get_wind_dir_idx()] = 1               # set the right category in the wind_dir vector

    windspeed_vec[self.model.windspeed] = 1

    # check output
    #print(self.model.windspeed)
    #print("wind: ", wind_dir_vec, windspeed_vec)
    #print("agent, wp len: ", len(agent_pos_with_waypoints))

    image_and_wind = [image, wind_dir_vec, windspeed_vec, agent_pos_with_waypoints] # this is one raw datapoint
    self.episode_data.append(image_and_wind)


  def append_episode(self):
    print("appending episode")
    self.all_data.extend(self.episode_data)
    self.episode_data.clear()


  def discard_episode(self):
    print("discarding episode")
    self.episode_data.clear()                               # ignore unsuccessful episode (BACKSPACE)


  def save_training_run(self):
    """Save all data to numpy files existing .npy
       files and the globals to .txt files. Triggered on
       closing the window. You need to provide a filename without
       extension!"""
    print("saving the run")

    if not len(self.all_data) > 0:
      print("no data gathered, not saving the run")
      return
    if self.name is None:
      self.name = input("full filename without .npy:")

    print("amount of datapoints saved: ", len(self.all_data))
    all_data = np.asarray(self.all_data,                    # object type because dimensions of picture
                          dtype=object)                     # and wind vectors are different
    print(np.shape(all_data))

    sep = os.path.sep                                       # cross-platform
    dirname = os.path.dirname(os.path.realpath(__file__)) + sep + ".." + sep + "data" + sep
    filename = dirname + "runs" + sep + self.name + ".npy"

    np.save(filename, all_data, allow_pickle=True)

    # another file with the same number, saving all relevant globals
    globals_file = open(dirname + "globals" + sep + self.name + ".txt", "w+")
    globals_file.write("# training examples: " + str(len(all_data)) + "size: " + str(size) + " #agents: " + str(nr_of_agents) + " timeframe: " + str(timeframe)
                       + " agentRadius: " + str(agentRadius) + " randseed: " + str(randseed))


  def get_wind_dir_idx(self):
    """Order of wind directions:
       N, S, E, W, NE, NW, SE, SW"""
    wind_dir = self.model.wind_dir
    if wind_dir == (Direction.NORTH, Direction.NORTH):
      return 0
    if wind_dir == (Direction.SOUTH, Direction.SOUTH):
      return 1
    if wind_dir == (Direction.EAST, Direction.EAST):
      return 2
    if wind_dir == (Direction.WEST, Direction.WEST):
      return 3
    if wind_dir == (Direction.NORTH, Direction.EAST):
      return 4
    if wind_dir == (Direction.NORTH, Direction.WEST):
      return 5
    if wind_dir == (Direction.SOUTH, Direction.EAST):
      return 6
    if wind_dir == (Direction.SOUTH, Direction.WEST):
      return 7