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

    datapoint = np.zeros((self.model.size, self.model.size), dtype=np.uint8)

    #  set NodeState to be pixel value:
    #  NORMAL = 0
    #  FIREBREAK = 1
    #  ON_FIRE = 2
    #  BURNED_OUT = 3
    #  AGENT = 4

    for y, node_row in enumerate(self.model.nodes):
      for x, node in enumerate(node_row):
        datapoint[y][x] = int(node.state)

    print(datapoint)
    exit()
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

  def save_episode(self):
    """Save all data to numpy files / extend existing .npy files"""
    pass


  def discard_episode(self):
    pass


  # should have an option to open npy files and append data points to them while playing