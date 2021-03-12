from collections import deque
from math import sqrt

from Model.direction import Direction
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

    # TODO M agent time step
    # IF timestep == a decision timestep (every 5-10 model time steps, first one should ofc be at time == 0)
      # change color
      # wait for human to give waypoint (probably good to print agent pos + "waiting for input" to cmd)
      # set new private waypoint
      # save state of the env, maybe make a new class that is responsible for that, so on exit we know where all the data is
    # ELSE
      # walk towards waypoint (dumb / A-Star) + dig on every step


    self.move()
    if self.save_move:
      self.agent_hist.append((self.position))

  def dig(self):
    self.model.dig_firebreak(self)


  def move(self):
    waypoint = self.get_waypoint() # M should be changed to "I got a waypoint assigned by a human, I will move in this direction"
    if waypoint is None:
      self.dig()
      return

    self.position = Direction.find(self.position, waypoint)(self.position)
    # self.position = Direction.DOWN(self.position)


  def get_waypoint(self):
    if len(self.model.waypoints) == 0:
      return None

    waypoints = list(self.model.waypoints)
    waypoint = waypoints[0]
    distance = self.distance_to(waypoint)
    if len(waypoints) > 1:
      for possible_waypoint in waypoints[1:]:
        new_distance = self.distance_to(possible_waypoint)
        if new_distance < distance:
          waypoint = possible_waypoint
          distance = new_distance

    return waypoint


  def distance_to(self, position):
    x, y = position
    return sqrt((x - self.position[0])**2 + (y - self.position[1])**2)

