import numpy as np

class DataSaver:
  def __init__(self, model):
    self.model = model


  def prepare_datapoint(self):
    """Should get called when new agent waypoints were set and model is about to
       fast-forward 5-10 timesteps to see how it played out."""
    pass
