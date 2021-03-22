from Model.agent import Agent
from Model.direction import Direction
from Model.data_saver import DataSaver
from Model.node import Node, NodeType
from enum import Enum
import random

# For data generation maybe lose the seed
random.seed(1)

class State(Enum):
  ONGOING = 0
  FIRE_CONTAINED = 1
  FIRE_OUT_OF_CONTROL = 2


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
    self.init_nodes()
    self.reset_necessary = False

    ## Fire initialization
    self.firesize = firesize
    self.firepos = set()
    self.start_episode()           # Initialize episode

    ## Data saving initialization
    self.DataSaver = DataSaver(self)


  def init_nodes(self):
    self.nodes = []
    for x in range(self.size):
      node_row = []
      for y in range(self.size):
        newNode = Node(self, (x, y), NodeType.GRASS, self.wind_dir)
        node_row.append(newNode)
      self.nodes.append(node_row)
    
    ## Set the neighbours
    for node_row in self.nodes:
      for node in node_row:
        pos = node.position
        neighbour = {"N": Direction.GO_NORTH(pos), "E": Direction.GO_EAST(pos),       ## Neighbours
                     "S": Direction.GO_SOUTH(pos), "W": Direction.GO_WEST(pos)}

        node.set_neighbours(north= self.find_node(neighbour["N"]),
                            east = self.find_node(neighbour["E"]),
                            south= self.find_node(neighbour["S"]),
                            west = self.find_node(neighbour["W"]))


## Episode Initialization
  def start_episode(self):
    self.reset_agents()
    self.waypoints = set() # Reset selection # M TODO are those the same as in Agent - are they updated?
    self.time = 0                 # Reset time
    self.state = State.ONGOING

    for node_row in self.nodes:
      for node in node_row:
        node.reset()

    # Start fire in the middle of the map
    self.firepos.clear()
    self.set_initial_fire(self.firesize)
    self.firebreaks = set()



  def set_initial_fire(self, firesize):
    ##TODO if using firesize to start with a larger fire, ignite some neighbours
    x = y = int(self.size / 2)
    centre_node = self.find_node((x, y))
    centre_node.ignite()


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
        agent.assign_new_waypoint() # waypoint to

    self.time += 1                # fire and agents need this info
    # 2
    #for agent in self.agents:
      #agent.timestep()            # walks 1 step towards current waypoint & digs on the way
    for node_row in self.nodes:
      for node in node_row:
        node.time_step()
    
    for node_row in self.nodes:
      for node in node_row:
        node.update_state()

    # 3

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


  def find_node(self, pos):
    if not self.position_in_bounds(pos):
      return None
    return self.nodes[pos[0]][pos[1]]


## TODO updated for the new nodes with boundary conditions
## Position management
  def get_neighbours(self, position):
    return {"N": self.find_node(Direction.GO_NORTH),
            "E": self.find_node(Direction.GO_EAST),
            "S": self.find_node(Direction.GO_SOUTH),
            "W": self.find_node(Direction.GO_WEST)}


  def agent_positions(self):
    return [agent.position for agent in self.agents]


  def get_random_position(self):
    return random.randint(0, self.size - 1), random.randint(0, self.size - 1)


  def is_firebreak(self, position):
    return position in self.firebreaks


  def position_in_bounds(self, position):
    x, y = position
    return 0 <= x < self.size and 0 <= y < self.size

  
## Model manipulation
  def select_square(self, position):
    for agent in self.agents:
      if position == agent.position:
        agent.dig()
        return

    if position not in self.firepos:       ## Cannot set waypoint in the fire
      self.waypoints.add(position)         # Add to list of waypoints


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

