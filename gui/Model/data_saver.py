import glob
import platform
from pathlib import Path, PureWindowsPath
import numpy as np
import os

class DataSaver:
  def __init__(self, model):
    self.model = model
    # TODO replace with numpy equivalents
    self.episode_data = list()  # gathered during episode, not sure yet whether it will succeed
    self.all_data = list()      # only data points leading to fire containment here

  def append_datapoint(self):
    """Should get called when new agent waypoints were set and model is about to
       fast-forward 5-10 timesteps to see how it played out."""

    datapoint = np.zeros((self.model.size, self.model.size), dtype=np.uint8)

    #  set NodeState to be pixel value:
    #  NORMAL = 0
    #  FIREBREAK = 1
    #  ON_FIRE = 2
    #  BURNED_OUT = 3
    #  AGENT = 4

    for y, node_row in enumerate(self.model.nodes): # make a picture of the node state
      for x, node in enumerate(node_row):
        datapoint[y][x] = node.state

    # wind direction?
    #picture_and_wind = [datapoint, model.wind_dir]

    self.episode_data.append(datapoint)


    # TO BE SAVED
    # agent pos + waypoints
    # fire pos
    # maybe tree pos
    # maybe wind dir



  def append_episode(self):
    self.all_data.extend(self.episode_data) # simply add
    self.episode_data.clear()
    print(len(self.all_data))
    print("appending episode")

  def discard_episode(self):
    print("discarding episode")
    self.episode_data.clear()               # ignore unsuccessful episode


  def save_training_run(self):
    """Save all data to numpy files / extend existing .npy files"""
    print("saving the run")
    print(len(self.all_data))
    all_data = np.asarray(self.all_data, dtype=np.uint8)
    print(np.shape(all_data))
    if platform.system() == 'Windows':
      print("using win")
      filenames = []
      for file in glob.glob(os.path.dirname(os.path.realpath(__file__)) + "\\..\\data\\*.npy"):
        filenames.append(file)

      filenames.sort()
      if not filenames:
        next_file_number = 0
      else:
        next_file_number = int(filenames[-1][-5]) + 1  # gets the X from 'runX.npy'

      datafolder = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
      filename = datafolder + "test" + str(next_file_number) + ".npy"
      print(filename)
    else:
      filenames = []
      for file in glob.glob(os.path.dirname(os.path.realpath(__file__)) + "/../data/*.npy"):
        filenames.append(file)

      filenames.sort()
      if not filenames:
        next_file_number = 0
      else:
        next_file_number = int(filenames[-1][-5]) + 1     # gets the X from 'runX.npy'


      datafolder = os.path.dirname(os.path.realpath(__file__)) + "/../data/"
      filename = datafolder + "test" + str(next_file_number) + ".npy"
    np.save(filename, all_data, allow_pickle=True)


  # should have an option to open npy files and append data points to them while playing