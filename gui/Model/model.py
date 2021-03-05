from Model.agent import Agent

class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, agents):
    self.size = length
    self.agents = agents
    self.firebreaks

    for agent in agents:
      agent.model = self

    self.startEpisode()           # Initialize episode

  
  def startEpisode(self):
    self.selected_squares = set() # Reset selection
    self.time = 0                 # Reset time
    self.terminal_state = False   # Restart so at initial state

    self.centreX = int(self.size / 2)
    self.centreY = int(self.size / 2)
    # Start fire in the middle of the map
    self.firepos = set()
    self.firepos.add((self.centreX, self.centreY))


  ## Increment the time by one.
  # TODO: possibly reset selection?
  def time_step(self):
    self.time += 1                # Increment time
    self.expand_fire()            # Determine fire propagation
    self.selected_squares = set()


  def select_square(self, position):
    if position not in self.firepos:       ## Cannot set waypoint in the fire
      self.selected_squares.add(position)  # Add to list of waypoints
    

  def deselect_square(self, position):
    self.selected_squares.discard(position) # Remove from list of waypoints
    
  def getNeighbours(self, position):
    self.pos = position
    x = self.pos[0]
    y = self.pos[1]
    if(x > 0 & x < self.size):
      left = x - 1
      right = x + 1
    if(y > 0 & y < self.size):
      top = y + 1
      bottom = y - 1
    ## new list of neighbouring cells
    neighbours = [(left, y), (x, top), (right, y), (x, bottom)]
    return neighbours

  ## currently stops when the fire reaches the edge of the map for simplicity but
  ## also as it makes it impossible for the agent to contain the fire
  def expand_fire(self):
    fire = list(self.firepos)
    for pos in fire:
      neighbours = self.getNeighbours(pos)
      for neighbour in neighbours:
        self.firepos.add(neighbour)


  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    print("Shutting Down")