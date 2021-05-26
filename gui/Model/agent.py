from collections import deque
from math import sqrt

from Model.direction import Direction
from Model.utils import *

from numpy import arctan2


class Agent:
  def __init__(self, position, model):
    self.position = position
    self.start_pos = position                               # to move in a straight line
    self.node = model.find_node(position)
    self.prev_node = model.find_node(position)              # old position needed to move in straight line to wp
    self.dead = False
    self.original_waypoint = None
    self.wp = None
    self.wp_digging = None
    self.wp_driving = None
    self.model = model
    self.is_digging = True


  def timestep(self):
    """Either dig+drive or drive 2 times based on waypoint type."""
    if self.is_digging:
      self.dig()
      self.move()                                           # walk towards waypoint (straight line) + dig on every step

    else:
      self.move()                                           # move at double the speed compared to digging
      self.move()


  def assign_new_waypoint(self, position, digging):
    """Takes a position selected by a mouse click and projects it onto
       a square (rotated diamond-like) around the agent, where it can
       actually reach it within X timesteps. This is done to have uniform
       waypoint distances from the agents."""
    self.original_waypoint = position

    self.is_digging = digging
    self.start_pos = self.position                          # save where the agent came from

    delta_x = position[0] - self.position[0]
    delta_y = position[1] - self.position[1]

    total_steps = abs(delta_x) + abs(delta_y)               # total steps cannot be bigger than timeframe for agents to move

    if self.is_digging:                                     # selected node closer to agent, then only walk/dig that far
      if total_steps < timeframe:
        self.wp_digging = position
        self.wp = position
        self.wp_driving = None
        return
    else:
      if total_steps < timeframe * 2:
        self.wp_digging = None
        self.wp = position
        self.wp_driving = position
        return

    if total_steps == 0:
      move_x = 0
      move_y = 0

    else:
      move_x = (delta_x / total_steps ) * (timeframe - 1)   # make use of the fact that we deal with 'similar triangles'
      move_y = (delta_y / total_steps) * (timeframe - 1)

    self.wp = [round(self.position[0] + move_x), round(self.position[1] + move_y)]

    if self.is_digging:
      self.wp_digging = self.wp
      self.wp_driving = None
    else:
      self.wp_digging = None
      self.wp_driving = [round(self.position[0] + 2 * move_x), round(self.position[1] + 2 * move_y)] # walking twice as fast


  def dig(self):
    self.node.dig_firebreak()

  def set_on_fire(self):
    self.dead = True

  def move(self):
    """Find the direction to go to, move one step N/S/E/W."""
    if self.wp_digging is None and self.wp_driving is None:
      return
    self.prev_node = self.node
    self.position = Direction.find(self)(self.position)
    self.node = self.model.find_node(self.position)
    self.model.agent_moves(self)


  def get_waypoint(self):
    """NOT USED"""
    print("wp")
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
    """NOT USED"""
    return self.model.find_node(self.position)


  def distance_to(self, position):
    x, y = position
    return sqrt((x - self.position[0])**2 + (y - self.position[1])**2)


  def angle(self):
    position_rel_to_centre =  self.position[0]- self.model.centre[0], \
                              self.position[1] - self.model.centre[1]
    
    return arctan2(position_rel_to_centre[1], position_rel_to_centre[0])



