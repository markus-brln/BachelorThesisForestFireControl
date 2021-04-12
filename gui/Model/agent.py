from collections import deque
from math import sqrt

from Model.direction import Direction
import random

random.seed(0)

class Agent:
  # TODO: Original code gave W as parameter. Find out purpose
  def __init__(self, position, model, active = False):
    self.position = position
    self.start_pos = position                 # to move in a straight line
    self.node = model.find_node(position)
    self.prev_node = model.find_node(position)
    self.dead = False
    self.digging = True
    self.active = active
    self.agent_hist = deque()
    self.agent_hist.append(self.position)
    self.save_move = False
    self.waypoint = None
    self.waypoint_old = None
    self.model = model

  def timestep(self):
    # M walk towards waypoint (dumb / A-Star) + dig on every step
    self.move()
    if self.save_move:
      self.agent_hist.append(self.position)

  def assign_new_waypoint(self, position):
    """Takes a position selected by a mouse click and projects it onto
       a square (rotated diamond-like) around the agent, where it can
       actually reach it within X timesteps. This is done to have uniform
       waypoint distances from the agents."""
    delta_x = position[0] - self.position[0]
    delta_y = position[1] - self.position[1]

    total_steps = abs(delta_x) + abs(delta_y)     # total steps cannot be bigger than timeframe for agents to move (10)
    timeframe = 10
    move_x = (delta_x / total_steps ) * (timeframe - 1) # make use of the fact that we deal with 'similar triangles'
    move_y = (delta_y / total_steps) * (timeframe - 1)

    self.waypoint = [round(self.position[0] + move_x), round(self.position[1] + move_y)]

    self.start_pos = self.position
    self.waypoint = position


  def dig(self):
    self.model.dig_firebreak(self)


  def set_on_fire(self):
    self.dead = True


  def move(self):
    if self.waypoint is None:
      return
    self.prev_node = self.node
    self.position = Direction.find(self)(self.position)
    self.node = self.model.find_node(self.position)
    self.node.dig_firebreak()
    self.model.agent_moves(self)


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

  def get_node(self):
    return self.model.find_node(self.position)


  def distance_to(self, position):
    x, y = position
    return sqrt((x - self.position[0])**2 + (y - self.position[1])**2)

