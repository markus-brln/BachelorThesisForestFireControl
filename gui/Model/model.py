from Model.agent import Agent

class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, agents, firesize = 1):
    self.size = length
    self.agents = agents

    ## Fire initialization
    self.firepos = set()
    self.set_initial_fire(firesize)

    self.startEpisode()           # Initialize episode

  
  def startEpisode(self):
    self.selected_squares = set() # Reset selection
    self.time = 0                 # Reset time
    self.terminal_state = False   # Restart so at initial state

    # Start fire in the middle of the map
    self.firepos.clear()
    self.firepos = set(self.initial_fire)


  def set_initial_fire(self, firesize):
    self.centre = [(int(self.size / 2), int(self.size / 2))]
    self.initial_fire = set(self.centre)
    for _ in range(firesize - 1):
      fire = list(self.initial_fire)
      for pos in fire:
        neighbours = self.getNeighbours(pos)
        for neighbour in neighbours:
          self.initial_fire.add(neighbour)


  ## Increment the time by one.
  # TODO: possibly reset selection?
  def time_step(self):
    self.time += 1                # Increment time
    self.expand_fire()            # Determine fire propagation


  def select_square(self, position):
    if position not in self.firepos:       ## Cannot set waypoint in the fire
      self.selected_squares.add(position)  # Add to list of waypoints
    

  def deselect_square(self, position):
    self.selected_squares.discard(position) # Remove from list of waypoints


  def getNeighbours(self, position):
    self.pos = position
    x = self.pos[0]
    y = self.pos[1]
    self.left = x - 1
    self.right = x + 1
    self.top = y + 1
    self.bottom = y - 1
    ## new list of neighbouring cells
    neighbours = [(self.left, y), (x, self.top), (self.right, y), (x, self.bottom)]
    for neighbour, (a, b) in enumerate(neighbours):
     if self.position_in_bounds((a, b)):
      return neighbours
     else:
      self.shut_down()


  ## currently stops when the fire reaches the edge of the map for simplicity but
  ## also as it makes it impossible for the agent to contain the fire
  def expand_fire(self):
    self.fire = list(self.firepos)
    for pos in self.fire:
      neighbours = self.getNeighbours(pos)
      for neighbour in neighbours:
        self.firepos.add(neighbour)

  def position_in_bounds(self, position):
    self.pos = position
    self.x = self.pos[0]
    self.y = self.pos[1]
    if ((self.x >= 0 & self.x <= self.size) & (self.y >= 0 & self.y <= self.size)):
      return True
    else:
      return False


  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    print("Fire out of control!")
    self.startEpisode()