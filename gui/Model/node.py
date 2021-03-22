from enum import Enum
from Model.direction import Direction

import random

random.seed(0)

## TODO? add colour
class NodeType (Enum):
# List:  [Fuel, Temperature, ignition_threshold]
  GRASS = 0
  WATER = 1


class NodeState (Enum):
  NORMAL = 0
  FIREBREAK = 1
  ON_FIRE = 2
  BURNED_OUT = 3


class Node:
  def __init__(self, environment, position, node_type, wind_dir):
    ## For callbacks
    self.environment = environment

    ## Initialization
    self.type = node_type
    self.set_default_properties()         ## Set fuel, temp and ignition threshold to default based on type
    self.state = NodeState.NORMAL
    self.next_state = NodeState.NORMAL
    self.position = position
    self.wind_dir = wind_dir

    ## Prepare for first episode
    self.reset()
  

  def reset(self):
    self.present_agent = None
    self.state = NodeState.NORMAL
    self.next_state = NodeState.NORMAL
    self.fuel = self.default_props["fuel"]
    self.temperature = self.default_props["temp"]
    self.ignition_threshold = self.default_props["ign_thres"]


  def set_default_properties(self):
    if self.type == NodeType.GRASS:
      self.default_props = {"fuel": 100, "temp": 0, "ign_thres": 2.5}
    if self.type == NodeType.WATER:
      self.default_props = {"fuel": 0, "temp": 0, "ign_thres": float("inf")}
  

  def set_neighbours(self, north, east, south, west):
    self.neighbours = {Direction.NORTH: north, 
                       Direction.EAST: east,
                       Direction.SOUTH: south,
                       Direction.WEST: west}

  
  def time_step(self):
    self.state = self.next_state
    self.next_state = self.state

    if self.state == NodeState.ON_FIRE:
      if self.present_agent != None:
        self.present_agent.set_on_fire()
        
      self.fuel -= 1
      if self.fuel <= 0:
        self.burn_out()
      
      self.heat_up_neighbours()

    # else:
      # self.temperature -= 0.25 


  ## TODO: Implement wind direction
  def heat_up_neighbours(self):
    for direction, neighbour in self.neighbours.items():
      if neighbour is not None:
        heat_spread = random.uniform(0.5, 1.5)
        if Direction.is_opposite(direction, self.wind_dir):
          heat_spread /= 2
        elif direction == self.wind_dir:
          heat_spread *= 2

        neighbour.heat_up(heat_spread)


  def heat_up(self, heat):
    self.temperature += heat
    if self.temperature >= self.ignition_threshold:
      self.ignite()


  def dig_firebreak(self):
    if self.state == NodeState.NORMAL:
      self.ignition_threshold = float("inf")
      self.next_state = NodeState.FIREBREAK

    self.environment.firebreak.add(self.position)


  def ignite(self):
    if self.fuel != 0 and not self.state == NodeState.ON_FIRE:
      self.next_state = NodeState.ON_FIRE
    
    self.environment.firepos.add(self.position) ## Let environment know you're on fire


  def burn_out(self):
    if self.state == NodeState.ON_FIRE:
      self.next_state = NodeState.BURNED_OUT
      self.environment.firepos.remove(self.position)

    
