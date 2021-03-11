from collections import deque
from math import sqrt

from enum import Enum

import random

random.seed(0)

class Direction(Enum):
  NONE = lambda pos: pos
  UP = lambda pos: (pos[0], pos[1] - 1)
  RIGHT = lambda pos: (pos[0] + 1, pos[1])
  DOWN = lambda pos: (pos[0], pos[1] + 1)
  LEFT = lambda pos: (pos[0] - 1, pos[1])

  @staticmethod
  def find(coming_from, going_to):
    if coming_from == going_to:
      return Direction.NONE

    delta_x = going_to[0] - coming_from[0]
    delta_y = going_to[1] - coming_from[1]
    if abs(delta_x) > abs(delta_y):
      return Direction.LEFT if delta_x < 0 else Direction.RIGHT
    return Direction.UP if delta_y < 0 else Direction.DOWN


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
    self.model.dig_firebreak(self)


  def move(self):
    waypoint = self.get_waypoint()
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

