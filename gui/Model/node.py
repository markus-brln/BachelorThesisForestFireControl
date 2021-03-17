

class Node:
  def __init__(self, position, fuel, temperature, ignition_threshold, neighbours, wind):
    self.position = position
    self.fuel = fuel
    self.temperature = temperature
    self.ignition_threshold = ignition_threshold
    self.neighbours = neighbours
    self.wind = wind

    self.on_fire = False
    self.present_agent = None

  
  def time_step(self):
    if self.on_fire:
      if self.present_agent != None:
        self.present_agent.set_on_fire()
        
      self.fuel -= 1
      if self.fuel <= 0:
        self.on_fire = False

    else:
      self.temperature -= 1


  def heat_up_neighbours(self):
    for neighbour in self.neighbours:
      if neighbour is not None:
        neighbour.heat_up(1)


  def heat_up(self, heat):
    self.temperature += heat
    if self.temperature >= self.ignition_threshold:
      self.ignite()


  def ignite(self):
    if self.fuel is not 0:
      self.on_fire = True

