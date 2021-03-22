from Model.agent import Agent
from Model.direction import Direction
from Model.data_saver import DataSaver
from Model.node import Node
from enum import Enum
import random

# For data generation maybe lose the seed
random.seed(1)

class State(Enum):
  ONGOING = 0
  FIRE_CONTAINED = 1
  FIRE_OUT_OF_CONTROL = 2

class WindDir(Enum):
  NORTH = 0
  SOUTH = 1
  EAST = 2
  WEST = 3


class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, nr_of_agents: int, firesize: int = 1, wind_dir = None):
    ## properties of env
    self.size = length
    self.nr_of_agents = nr_of_agents
    self.wind_dir = wind_dir

    ## initial properties of this model
    self.agents = []
    self.state = State.ONGOING
    self.nodes = list()           # all nodes of the NxN grid
    self.init_nodes()
    self.reset_necessary = False

    ## Fire initialization
    self.firesize = firesize
    self.firepos_edge = list()    # view will only fires new on the edge (not surrounded by fire)
    self.firepos = set()          # unique fire cells
    self.start_episode()          # Initialize episode

    ## Data saving initialization
    self.DataSaver = DataSaver(self)


  def init_nodes(self):
    ## arbitrary values for initialisation
    fuel = 10
    temp = 0
    thresh = 2.5
    for x in range(self.size):
      for y in range(self.size):
                        ## position, fuel, temperature, ignition_threshold, neighbours, wind
        newNode = Node((x, y), fuel, temp, thresh, self.get_neighbours((x, y)), self.wind_dir)
        # print("pos: ", newNode.position, "temp: ", newNode.temperature, "thresh: ", newNode.ignition_threshold)
        # if(newNode.position == ((int(self.size / 2), int(self.size / 2)))):
        #   newNode.ignite()
        self.nodes.append(newNode)

## Episode Initialization
  def start_episode(self):
    self.reset_agents()
    self.waypoints = set() # Reset selection # M TODO are those the same as in Agent - are they updated?
    self.time = 0                 # Reset time
    self.state = State.ONGOING

    # Start fire in the middle of the map
    self.firepos.clear()
    self.firepos_edge.clear()
    self.set_initial_fire(self.firesize)
    self.firebreaks = set()



  def set_initial_fire(self, firesize):
    ##TODO if using firesize to start with a larger fire, ignite some neighbours
    x = y = int(self.size / 2)
    centre_node = self.find_node((x, y))
    centre_node.ignite()
    self.firepos.add(centre_node.position)
    self.firepos_edge.append(centre_node.position)


  def find_node(self, pos):
    x, y = pos
    for node in self.nodes:
      if node.position == (x, y):
        return node
    return


  def quadrants(self):
    n = self.size
    m = 0
    for x in range(0, int(self.size / 2)):
      for y in range(n, m):
        self.west.append(x, y)

  
  # Start agents at random positions
  def reset_agents(self):
    self.agents.clear()
    for _ in range(self.nr_of_agents):
      agent_pos = self.get_random_position()
      while not self.position_in_bounds(agent_pos) or agent_pos in list(self.firepos):
        agent_pos = self.get_random_position()
      self.agents += [Agent(agent_pos, self)]




## Time propagation
  # TODO: possibly reset selection?
  def time_step(self):
    """Order of events happening during time step:
    0. save old agent waypoints every 5-10 time steps (not at start ofc)
    1. set agent waypoints (at start + every 5-10 waypoints)
    2. agent time step (move towards waypoint)
    3. expand fire
    4. check whether fire out of control (restart episode if necessary)
    5. check if fire contained (save episode + restart)
    """
    # 0
    # if self.time % 5 == 0 and self.time != 0: # self.time != 0 meaning self.agents[0].waypoint_old is not None
    #       self.DataSaver.append_datapoint()

    # 1
    if self.time % 5 == 0:        # every 5 time steps new waypoints should be set
      #print("agents require new waypoints")
      for agent in self.agents:
        agent.assign_new_waypoint()

    self.time += 1                # fire and agents need this info
    # 2
    #for agent in self.agents:
      #agent.timestep()            # walks 1 step towards current waypoint & digs on the way

    # 3
    ## fire expands every 3 timesteps
    #if self.time % 3 == 0:
    self.expand_fire()            # Determine fire propagation

    #print("expanding fire took: ", timelib.time() - start)
    # print(len(self.firepos))

    # 4
    # self.waypoints.clear() # Reset selection
    if self.state == State.FIRE_OUT_OF_CONTROL:
      # M do not safe the gathered data points of the episode
      self.start_episode()
      self.reset_necessary = True     # M view needs to be updated by controller... not a nice way but works
      return

    # 5
    # IF fire contained
      # self.DataSaver.append_episode()
      # self.start_episode()
      # return  # I guess it has to return to start new



  ## currently stops when the fire reaches the edge of the map for simplicity but
  ## also as it makes it impossible for the agent to contain the fire
  def expand_fire(self):
    """Expands every x time steps by simply igniting its neighbours.
       1. Look for non-fire neighbour cells not in self.firepos
       2. add new fires to self.firepos
       3. update self.firepos_edge
    """

    #firepos_new_tmp = set()                     # unique new neighbours
    agent_positions = self.agent_positions()    # call function here (only once)

    # 1. Look for non-fire neighbour cells not in self.firepos
    for pos in self.firepos_edge:
      neighbours = self.get_neighbours_not_on_fire(pos)
      for neighbour, direction in zip(neighbours, range(len(neighbours))):
        #if neighbour in agent_positions:
        #  print("Fire took out agent. Restarting episode.")
        #  self.state = State.FIRE_OUT_OF_CONTROL
        #  return
        if not self.position_in_bounds(neighbour):
          print("Fire exceeded boundaries. Restarting episode.")
          self.state = State.FIRE_OUT_OF_CONTROL
          return
        if not self.is_firebreak(neighbour):
          # 2. add new fires to self.firepos
          self.firepos.add(neighbour)
        else:
          print("can't expand through firebreak @", neighbour)

    # 3. update self.firepos_edge
    self.firepos_edge.clear()
    self.firepos_edge = self.find_edge_fires(list(self.firepos))



  def find_edge_fires(self, firepos):
    edge_fires = set()

    for pos in firepos:
      neighbours = self.get_neighbours(pos)
      surrounded = True
      for neighbour in neighbours:
        if neighbour not in firepos:
          surrounded = False

      if not surrounded:
        edge_fires.add(pos)

    return list(edge_fires)


## TODO updated for the new nodes with boundary conditions
## Position management
  def get_neighbours(self, position):
    x, y = position
            #   NORTH         SOUTH     EAST         WEST
    return [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]

  def get_neighbours_not_on_fire(self, position):
    """Looks whether the N, S, E, W neighbours are already on fire.
    @:return list of (x,y) neighbours to set on fire
    """
    x, y = position
    new_neighbours = []

    if (x, y - 1) not in self.firepos:
      if self.wind_dir == WindDir.SOUTH:
        new_neighbours.append((x, y - 1))   # NORTH
      else:
        if random.randint(0,1) != 0:
          new_neighbours.append((x, y - 1))

    if (x, y + 1) not in self.firepos:
      if self.wind_dir == WindDir.NORTH:
        new_neighbours.append((x, y + 1))   # SOUTH
      else:
        if random.randint(0,1) != 0:
          new_neighbours.append((x, y + 1))
    if (x + 1, y) not in self.firepos:
      if self.wind_dir == WindDir.WEST:
        new_neighbours.append((x + 1, y))   # EAST
      else:
        if random.randint(0,1) != 0:
          new_neighbours.append((x + 1, y))
    if (x - 1, y) not in self.firepos:
      if self.wind_dir == WindDir.EAST:
        new_neighbours.append((x - 1, y))   # WEST
      else:
        if random.randint(0,1) != 0:
          new_neighbours.append((x - 1, y))

    return new_neighbours


  def agent_positions(self):
    return [agent.position for agent in self.agents]


  def get_random_position(self):
    return random.randint(0, self.size - 1), random.randint(0, self.size - 1)

  def is_firebreak(self, position):
    return position in self.firebreaks

  def position_in_bounds(self, position):
    x, y = position
    return 0 <= x < self.size and 0 <= y < self.size
    # M maybe like this, since fire not containable anymore when it already touches borders
    #return 1 <= x < self.size - 1 and 1 <= y < self.size - 1

  
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
    self.DataSaver.save_data()  # M data points of all successful episodes until here saved
    self.start_episode()

