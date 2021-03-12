import numpy as np

class DataSaver:
  def __init__(self, model):
    self.model = model
    # TODO replace with numpy equivalents
    self.episode_data = list()  # gathered during episode, not sure yet whether it will succeed
    self.all_data = list()      # only data points leading to fire containment here

  def append_datapoint(self):
    """Should get called when new agent waypoints were set and model is about to
       fast-forward 5-10 timesteps to see how it played out."""


    # TO BE SAVED
    # agent pos + waypoints
    # fire pos
    # maybe tree pos
    # maybe wind dir

    print("appending data point")
    data_point = []
    firepos = self.model.firepos


  def append_episode(self):
    self.all_data.extend(self.episode_data) # simply add

  def save_data(self):
    """Save all data to numpy files / extend existing .npy files"""
    pass


  # should have an option to open npy files and append data points to them while playing