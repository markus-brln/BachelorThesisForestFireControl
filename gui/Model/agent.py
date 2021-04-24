from collections import deque
from math import sqrt

from Model.direction import Direction
from Model.utils import *

from numpy import arctan2


class Agent:
  # TODO: Original code gave W as parameter. Find out purpose
  def __init__(self, position, model, active = False):
    self.position = position
    self.start_pos = position                 # to move in a straight line
    self.node = model.find_node(position)
    self.prev_node = model.find_node(position)
    self.dead = False
    self.active = active
    self.agent_hist = deque()
    self.agent_hist.append(self.position)
    self.save_move = False
    self.original_waypoint = None
    self.waypoint = None
    self.waypoint_digging = None
    self.waypoint_walking = None
    self.model = model
    self.is_digging = True

  def timestep(self, time):

    if self.is_digging:
      # M walk towards waypoint (dumb / A-Star) + dig on every step
      self.dig()
      #if time % 3 == 0:      # makes the simulation too slow unfortunately
      self.move()
      if self.save_move:
        self.agent_hist.append(self.position)
    else:
      self.move()               # move at double the speed compared to digging
      self.move()
      if self.save_move:
        self.agent_hist.append(self.position)

  def assign_new_waypoint(self, position, digging):
    """Takes a position selected by a mouse click and projects it onto
       a square (rotated diamond-like) around the agent, where it can
       actually reach it within X timesteps. This is done to have uniform
       waypoint distances from the agents."""
    self.original_waypoint = position

    self.is_digging = digging
    self.start_pos = self.position            # save where the agent came from

    delta_x = position[0] - self.position[0]
    delta_y = position[1] - self.position[1]


    total_steps = abs(delta_x) + abs(delta_y)     # total steps cannot be bigger than timeframe for agents to move

    if self.is_digging:                       # selected node closer to agent, then only walk/dig that far
      if total_steps < timeframe:
        self.waypoint_digging = position
        self.waypoint = position              # TODO get rid of just the waypoint (many things may depend on it still being here!)
        self.waypoint_walking = None
        return
    else:
      if total_steps < timeframe * 2:
        self.waypoint_digging = None
        self.waypoint = position
        self.waypoint_walking = position
        return



    if total_steps == 0:
      move_x = 0
      move_y = 0

    else:
      move_x = (delta_x / total_steps ) * (timeframe - 1) # make use of the fact that we deal with 'similar triangles'
      move_y = (delta_y / total_steps) * (timeframe - 1)

    self.waypoint = [round(self.position[0] + move_x), round(self.position[1] + move_y)]

    if self.is_digging:
      self.waypoint_digging = self.waypoint
      self.waypoint_walking = None
    else:
      self.waypoint_digging = None
      self.waypoint_walking = [round(self.position[0] + 2 * move_x), round(self.position[1] + 2 * move_y)] # walking twice as fast


    #self.waypoint = position # waypoints won't be on the position where we clicked


  def dig(self):
    # self.model.dig_firebreak(self)
    self.node.dig_firebreak()

  def set_on_fire(self):
    self.dead = True


  def move(self):
    if self.waypoint_digging is None and self.waypoint_walking is None:
      return
    self.prev_node = self.node
    self.position = Direction.find(self)(self.position)
    self.node = self.model.find_node(self.position)
    # self.node.dig_firebreak()
    self.model.agent_moves(self)


  def get_waypoint(self):
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
    return self.model.find_node(self.position)


  def distance_to(self, position):
    x, y = position
    return sqrt((x - self.position[0])**2 + (y - self.position[1])**2)
  

  def pos_rel_to_centre(self):
    return (self.position[0]- self.model.centre[0], self.position[1] - self.model.centre[1])


  def angle(self):
    position_rel_to_centre =  self.pos_rel_to_centre()
    
    return arctan2(position_rel_to_centre[1], position_rel_to_centre[0])



