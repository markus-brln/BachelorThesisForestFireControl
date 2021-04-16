from enum import Enum, IntEnum
from Model.direction import Direction
import numpy as np
import random

random.seed(0)

## TODO? add colour
class NodeType (Enum):
# List:  [Fuel, Temperature, ignition_threshold]
  GRASS = 0
  WATER = 1


class NodeState (IntEnum):
  NORMAL = 0
  FIREBREAK = 1
  ON_FIRE = 2
  BURNED_OUT = 3
  AGENT = 4


class Node:
  def __init__(self, environment, position, node_type, wind_dir, windspeed):
    ## For callbacks
    self.environment = environment
    (Direction.NORTH, 5) # TODO what is this?

    ## Initialization
    self.type = node_type
    self.position = position
    self.wind_dir = wind_dir
    self.windspeed = windspeed
    self.set_default_properties()         ## Set fuel, temp and ignition threshold to default based on type
    self.state = NodeState.NORMAL
    self.next_state = NodeState.NORMAL



    ## Prepare for first episode
    self.reset()


  def reset(self):
    self.present_agent = None
    self.fuel = self.default_props["fuel"]
    self.temperature = self.default_props["temp"]
    self.ignition_threshold = self.default_props["ign_thres"]
    self.next_state = NodeState.NORMAL
    self.update_state()


  def set_default_properties(self):
    if self.type == NodeType.GRASS:
      self.default_props = {"fuel": 20, "temp": 0, "ign_thres": 2}
    if self.type == NodeType.WATER:
      self.default_props = {"fuel": 0, "temp": 0, "ign_thres": float("inf")}
  

  def set_neighbours(self, north, east, south, west):
    self.neighbours = {Direction.NORTH: north, 
                       Direction.EAST: east,
                       Direction.SOUTH: south,
                       Direction.WEST: west}

  
  ## Time iteration 
  def time_step(self):
    if self.state == NodeState.ON_FIRE:
      if self.present_agent != None:
        self.present_agent.set_on_fire()
        
      self.fuel -= 1
      if self.fuel <= 0:
        self.burn_out()
      
      self.heat_up_neighbours()
  
  
  def heat_up_neighbours(self):
    for direction, neighbour in self.neighbours.items():
      if neighbour is not None:
        heat_spread = 1 #TODO Stochastic?
        if any(Direction.is_opposite(direction, wind_dir) for wind_dir in self.wind_dir):
          heat_spread /= (1 + (self.windspeed / 5))
        elif direction in self.wind_dir:
          if(self.windspeed > 0):
            heat_spread *= 1 + np.log(self.windspeed)
          else:
            heat_spread = 1
        neighbour.heat_up(heat_spread)


  def heat_up(self, heat):
    self.temperature += heat
    if self.temperature >= self.ignition_threshold:
      self.ignite()


  ## State changes
  def update_state(self):
    if self.state is not self.next_state:
      self.state = self.next_state
      self.environment.node_state_change(self)


  def dig_firebreak(self):
    if self.state != NodeState.ON_FIRE:
      self.ignition_threshold = float("inf")
      self.next_state = NodeState.FIREBREAK


  def ignite(self):
    if self.fuel != 0 and self.state == NodeState.NORMAL:
      self.next_state = NodeState.ON_FIRE


  def burn_out(self):
    if self.state == NodeState.ON_FIRE:
      self.next_state = NodeState.BURNED_OUT

  