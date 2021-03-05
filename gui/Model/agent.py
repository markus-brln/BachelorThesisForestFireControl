from collections import deque

class Agent:
  # TODO: Original code gave W as parameter. Find out purpose
  def __init__(self, position, active = False):
    self.x, self.y = position
    self.dead = False
    self.digging = True
    self.active = active
    self.agent_hist = deque()
    self.agent_hist.append((self.x, self.y))
    self.save_move = True
    self.waypoint = None
    self.model = None

  #
  def timestep(self):
    if (self.x, self.y) == self.waypoint:
      # TODO: Determine if this is appropriate
      self.dig()
    else:
      self.move()
      if self.save_move:
        self.agent_hist.append((self.x, self.y))

  def dig(self):
    pass

  def move(self):
    if self.waypoint == None:
      return
    # TODO: Implementation
    pass

  def set_waypoint(self, waypoint = None):
    self.waypoint = waypoint
