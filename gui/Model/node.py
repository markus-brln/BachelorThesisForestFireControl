

class Node:
  def __init__(self, position, fuel, temperature, ignition_threshold, neighbours):
    self.position = position
    self.fuel = fuel
    self.temperature = temperature
    self.ignition_threshold = ignition_threshold
    self.neighbours = neighbours
    self.on_fire = False
    self.present_agent = None

  
  def time_step(self):
    if self.on_fire:
      for neighbour in self.neighbours:
        neighbour.heat_up(1)

      self.fuel -= 1
      if self.fuel <= 0:
        self.on_fire = False

    else:
      self.temperature -= 1


  def heat_up(self, heat):
    self.temperature += heat
    if self.temperature >= self.ignition_threshold:
      self.ignite

  def ignite(self):
    if self.fuel is not 0:
      self.on_fire = True

