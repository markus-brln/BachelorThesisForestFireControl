import glob
import platform
from pathlib import Path, PureWindowsPath
import numpy as np
import os

from gui.Model.node import NodeState
from gui.Model.utils import *

class DataSaver:
  def __init__(self, model):
    self.model = model
    self.episode_data = list()  # gathered during episode, not sure yet whether it will succeed
    self.all_data = list()      # working with lists, later converted to np array

  def append_datapoint(self):
    """Should get called when new agent waypoints were set and model is about to
       fast-forward 5-10 timesteps (see utils.py, 'timeframe') to see how it played out."""

    picture = np.zeros((self.model.size, self.model.size), dtype=np.uint8) # save memory with uint8
    wind_dir_vec = np.zeros(n_wind_dirs, dtype=np.uint8)              # 8 wind directions
    windspeed_vec = np.zeros(n_wind_speed_levels, dtype=np.uint8)     # 5 'wind speed levels'

    #  set NodeState to be pixel value:
    #  NORMAL = 0
    #  FIREBREAK = 1
    #  ON_FIRE = 2
    #  BURNED_OUT = 3
    #  AGENT = 4
    for y, node_row in enumerate(self.model.nodes): # make a picture of the node state
      for x, node in enumerate(node_row):
        picture[y][x] = node.state
        #if node.state == NodeState.AGENT:
        #  agents[y][x] = 1
        #if node.state == NodeState.ON_FIRE:
        #  fire[y][x] = 1
        #if node.state == NodeState.FIREBREAK:
        #  firebreaks[y][x] = 1
        #if (y, x) in self.model.waypoints:
        #  waypoints[y][x] = 1

    # TODO create winddir and windspeed vectors (put exactly one 1 in them)

    picture_and_wind = [picture, wind_dir_vec, windspeed_vec]
    self.episode_data.append(picture_and_wind)


  def append_episode(self):
    print("appending episode")
    self.all_data.extend(self.episode_data) # simply add
    self.episode_data.clear()


  def discard_episode(self):
    print("discarding episode")
    self.episode_data.clear()               # ignore unsuccessful episode


  def save_training_run(self):
    """Save all data to numpy files existing .npy
       files and the globals to .txt files"""
    print("saving the run")
    if not len(self.all_data) > 0:
      print("no data gathered, not saving the run")
      return

    print("amount of datapoints saved: ", len(self.all_data))
    all_data = np.asarray(self.all_data, # object type because dimensions of picture
                          dtype=object)  # and wind vectors are different
    print(np.shape(all_data))

    sep = os.path.sep           # cross-platform
    dirname = os.path.dirname(os.path.realpath(__file__)) + sep + ".." + sep + "data" + sep

    filenames = []
    for file in glob.glob(dirname + "runs" + sep + "*.npy"):
      filenames.append(file)

    filenames.sort()
    if not filenames:
      next_file_number = 0
    else:
      number = str()
      for maybe_number in filenames[-1]:
        if maybe_number.isdigit():
          number = number + maybe_number
      next_file_number = int(number) + 1
      print(number)

    filename = dirname + "runs" + sep + "run" + str(next_file_number) + ".npy"

    np.save(filename, all_data, allow_pickle=True)

    # another file with the same number, saving all relevant globals
    globals_file = open(dirname + "globals" + sep + str(next_file_number) + ".txt", "w+")
    globals_file.write("size: " + str(size) + " #agents: " + str(nr_of_agents) + " windspeed: "
                       + str(windspeed) + " wind_dir: " + str(wind_dir) + " timeframe: " + str(timeframe)
                       + " agentRadius: " + str(agentRadius) + " randseed: " + str(randseed))


  # should have an option to open npy files and append data points to them while playing
  # M: not really necessary with a script that combines all files to a proper I/O file, otherwise we might risk
  #    screwing up a very big file