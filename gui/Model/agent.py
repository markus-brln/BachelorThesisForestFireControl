from collections import deque

import random

random.seed(0)

class Agent:
  # TODO: Original code gave W as parameter. Find out purpose
  def __init__(self, position, model, active = False):
    self.position = position
    self.dead = False
    self.digging = True
    self.active = active
    self.agent_hist = deque()
    self.agent_hist.append(self.position)
    self.save_move = False
    self.waypoint = None
    self.model = model 

  def timestep(self):
    self.move()
    if self.save_move:
      self.agent_hist.append((self.position))

  def dig(self):
    print("agent digging")
    self.model.dig_firebreak(self)

## Move randomly for now
  def move(self):
    new_position = random.choice(self.model.get_neighbours(self.position))
    while not self.model.position_in_bounds(new_position) or new_position in self.model.firepos:
      new_position = random.choice(self.model.get_neighbours(self.position))

    self.position = new_position
    

  def set_waypoint(self, waypoint = None):
    self.waypoint = waypoint
