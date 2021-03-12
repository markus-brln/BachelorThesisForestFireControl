from Model.agent import Agent
from Model.direction import Direction
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

    ## map coords
    self.north = list()
    self.east = list()
    self.south = list()
    self.west = list()
    self.quadrants()

    ## Fire initialization
    self.firepos = set()
    self.set_initial_fire(firesize)
    self.start_episode()           # Initialize episode




## Episode Initialization
  def start_episode(self):
    self.reset_agents()
    self.waypoints = set() # Reset selection
    self.time = 0                 # Reset time
    self.state = State.ONGOING

    # Start fire in the middle of the map
    self.firepos.clear()
    self.firepos = set(self.initial_fire)
    self.firebreaks = set()

  def quadrants(self):
    n = self.size
    m = 0
    for x in range(0, int(self.size / 2)):
      for y in range(n, m):
        self.west.append(x, y)

  
  # Start agents at random positions
  def reset_agents(self):
    self.agents = []
    for _ in range(self.nr_of_agents):
      agent_pos = self.get_random_position()
      while not self.position_in_bounds(agent_pos) or agent_pos in self.initial_fire:
        agent_pos = self.get_random_position()
      self.agents += [Agent(agent_pos, self)]


  def set_initial_fire(self, firesize):
    self.centre = [(int(self.size / 2), int(self.size / 2))]
    self.initial_fire = set(self.centre)
    for _ in range(firesize - 1):
      fire = list(self.initial_fire)
      for pos in fire:
        neighbours = self.get_neighbours(pos)
        for neighbour in neighbours:
          self.initial_fire.add(neighbour)


## Time propagation
  # TODO: possibly reset selection?
  def time_step(self):
    self.time += 1                # Increment time
    self.expand_fire()            # Determine fire propagation
    print(len(self.firepos))

    for agent in self.agents:     # M: each agent should change color to show that it's its turn to be assigned a new
      agent.timestep()            # waypoint to, then timesteps should pass where it walks to the waypoint & digs

    # self.waypoints.clear() # Reset selection
    if self.state == State.FIRE_OUT_OF_CONTROL:
      self.start_episode()
      return


  ## currently stops when the fire reaches the edge of the map for simplicity but
  ## also as it makes it impossible for the agent to contain the fire
  def expand_fire(self):
    ## fire expands 3 times more slowly than agents can move
    if self.time % 3 != 0:
      return
    self.fire = list(self.firepos)
    for pos in self.fire:
      neighbours = self.get_neighbours(pos)
      for neighbour in neighbours:
        if not self.position_in_bounds(neighbour):
          self.state = State.FIRE_OUT_OF_CONTROL
        if not self.is_firebreak(neighbour):
          self.firepos.add(neighbour)
          if neighbour in self.waypoints:
            self.waypoints.remove(neighbour)
        else:
          print("can't expand through firebreak @", neighbour)


## Position management
  def get_neighbours(position):
    x, y = position
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


  def agent_positions(self):
    return [agent.position for agent in self.agents]


  def get_random_position(self):
    return (random.randint(0, self.size - 1), random.randint(0, self.size - 1))

  def is_firebreak(self, position):
    return position in self.firebreaks

  def position_in_bounds(self, position):
    x, y = position
    return x >= 0 and x < self.size and y >= 0 and y < self.size

  
## Model manipulation
  def select_square(self, position):
    for agent in self.agents:
      if position == agent.position:
        agent.dig()
        return

    if position not in self.firepos:       ## Cannot set waypoint in the fire
      self.waypoints.add(position)  # Add to list of waypoints


  def deselect_square(self, position):
    self.waypoints.discard(position) # Remove from list of waypoints

  
  def dig_firebreak(self, agent):
    self.firebreaks.add(agent.position)
    if agent.position in self.waypoints:
      self.waypoints.remove(agent.position)


## Proper shutdown
  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    self.start_episode()

