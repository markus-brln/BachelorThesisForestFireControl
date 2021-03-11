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
    print("StartEpisode Called")
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
    fire_in_bounds = self.expand_fire()            # Determine fire propagation
    if not fire_in_bounds:
      self.startEpisode()

  def select_square(self, position):
    if position not in self.firepos:       ## Cannot set waypoint in the fire
      self.selected_squares.add(position)  # Add to list of waypoints
    

  def deselect_square(self, position):
    self.selected_squares.discard(position) # Remove from list of waypoints


  def getNeighbours(self, position):
    x, y = position
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

  ## currently stops when the fire reaches the edge of the map for simplicity but
  ## also as it makes it impossible for the agent to contain the fire
  def expand_fire(self):
    self.fire = list(self.firepos)
    for pos in self.fire:
      neighbours = self.getNeighbours(pos)
      for neighbour in neighbours:
        if not self.position_in_bounds(neighbour):
          return False
        
        self.firepos.add(neighbour)

    return True

  def position_in_bounds(self, position):
    x, y = position
    return x >= 0 and x <= self.size and y >= 0 and y <= self.size

  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    self.startEpisode()