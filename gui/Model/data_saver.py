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

    for y, node_row in enumerate(self.model.nodes): # make a picture of the node state
      for x, node in enumerate(node_row):
        datapoint[y][x] = node.state

    # wind direction?
    #picture_and_wind = [datapoint, model.wind_dir]

    self.episode_data.append(datapoint)

    print(len(self.episode_data))


    # TO BE SAVED
    # agent pos + waypoints
    # fire pos
    # maybe tree pos
    # maybe wind dir



  def append_episode(self):
    self.all_data.extend(self.episode_data) # simply add
    print(len(self.all_data))
    print("append episode")
    exit()

  def discard_episode(self):
    self.episode_data.clear()               # ignore unsuccessful episode


  def save_training_run(self):
    """Save all data to numpy files / extend existing .npy files"""
    pass

  # should have an option to open npy files and append data points to them while playing