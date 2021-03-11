from Model.agent import Agent
from enum import Enum

import random


# For data generation maybe lose the seed
random.seed(1)

class State(Enum):
  ONGOING = 0
  FIRE_CONTAINED = 1
  FIRE_OUT_OF_CONTROL = 2

class Model():
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, nr_of_agents: int, firesize: int = 1):
    self.size = length
    self.nr_of_agents = nr_of_agents

    ## Fire initialization
    self.firepos = set()
    self.set_initial_fire(firesize)

    self.startEpisode()           # Initialize episode

  
  def startEpisode(self):
    self.reset_agents()
    self.selected_squares = set() # Reset selection
    self.time = 0                 # Reset time
    self.terminal_state = State.ONGOING

    # Start fire in the middle of the map
    self.firepos.clear()
    self.firepos = set(self.initial_fire)

  
  # Start agents at random positions
  def reset_agents(self):
    self.agents = []
    for _ in range(self.nr_of_agents):
      agent_pos = self.random_position()
      while not self.position_in_bounds(agent_pos) or agent_pos in self.initial_fire:
        agent_pos = self.random_position()
      self.agents += [Agent(agent_pos)]



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
    if self.terminal_state == State.FIRE_OUT_OF_CONTROL:
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
          self.terminal_state = State.FIRE_OUT_OF_CONTROL
        
        self.firepos.add(neighbour)

  def random_position(self):
    return (random.randint(0, self.size - 1), random.randint(0, self.size - 1))


  def position_in_bounds(self, position):
    x, y = position
    return x >= 0 and x <= self.size and y >= 0 and y <= self.size


  def agent_positions(self):
    return [agent.position for agent in self.agents]


  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    self.startEpisode()