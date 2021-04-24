import math

from Model.agent import Agent
from Model.direction import Direction
from Model.data_saver import DataSaver
from Model.node import Node, NodeType, NodeState
from View.updatetype import UpdateType
from Model.utils import *
from enum import Enum
import random

# For data generation maybe lose the seed
from Model.utils import n_wind_speed_levels

#random.seed(1)


class State(Enum):
  ONGOING = 0
  FIRE_CONTAINED = 1
  FIRE_OUT_OF_CONTROL = 2


class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, nr_of_agents: int, radius: int):
    self.counter = 0
    self.firebreaks = set()
    self.waypoints = set()
    self.waypoints_walking = set()
    self.waypoints_digging = set()
    self.time = 0
    self.subscribers = []
    ## properties of env
    self.size = length
    self.centre = (int(length / 2), int(length / 2))
    self.nr_of_agents = nr_of_agents
    self.wind_dir = self.set_wind_dir()
    print(self.wind_dir)
    self.windspeed = self.set_windspeed()
    # self.windspeed = 5
    print("speed: ", self.windspeed)

    ## initial properties of this model
    self.agents = []
    self.nodes = []
    self.state = State.ONGOING
    self.init_nodes()
    self.reset_necessary = False

    ## Fire initialization
    self.radius = radius
    self.firepos = set()
    self.start_episode()  # Initialize episode

    ## Data saving initialization
    self.DataSaver = DataSaver(self)
    self.highlighted_agent_nr = None
    self.highlighted_agent = None

  ## Episode Initialization
  def start_episode(self):
    self.counter += 1
    print(f"{self.counter}th run")
    self.reset_agents()
    self.waypoints = set()
    self.waypoints_walking = set()
    self.waypoints_digging = set()
    self.time = 0
    self.state = State.ONGOING

    for node_row in self.nodes:
      for node in node_row:
        node.reset()

    # Start fire in the middle of the map
    self.firepos.clear()
    self.set_initial_fire(0)
    self.firebreaks = set()

    for subscriber in self.subscribers:
      subscriber.update(UpdateType.RESET)

  def init_nodes(self):
    for x in range(self.size):
      node_row = []
      for y in range(self.size):
        newNode = Node(self, (x, y), NodeType.GRASS, self.wind_dir, self.windspeed)
        node_row.append(newNode)
      self.nodes.append(node_row)

      ## Set the neighbours
    for node_row in self.nodes:
      for node in node_row:
        pos = node.position
        neighbour = {"N": Direction.GO_NORTH(pos), "E": Direction.GO_EAST(pos),  ## Neighbours
                     "S": Direction.GO_SOUTH(pos), "W": Direction.GO_WEST(pos)}

        node.set_neighbours(north=self.find_node(neighbour["N"]),
                            east=self.find_node(neighbour["E"]),
                            south=self.find_node(neighbour["S"]),
                            west=self.find_node(neighbour["W"]))

  def append_datapoint(self):
    print("Saving waypoints")
    self.DataSaver.append_datapoint()

  def discard_episode(self):
    self.DataSaver.discard_episode()

  def append_episode(self):
    self.DataSaver.append_episode()

  def save_training_run(self):
    self.DataSaver.save_training_run()

  def set_initial_fire(self, firesize):
    ##TODO if using firesize to start with a larger fire, ignite some neighbours
    x = y = int(self.size / 2)
    centre_node = self.find_node((x, y))
    centre_node.ignite()

  @staticmethod
  def set_windspeed():
    """values from 0 to including 4 is there are 5 wind speed levels"""
    return random.randint(0, n_wind_speed_levels-1)

  def set_wind_dir(self):
    wind_dirs = {0: (Direction.NORTH, Direction.NORTH),
                 1: (Direction.NORTH, Direction.EAST),
                 2: (Direction.EAST, Direction.EAST),
                 3: (Direction.SOUTH, Direction.EAST),
                 4: (Direction.SOUTH, Direction.SOUTH),
                 5: (Direction.SOUTH, Direction.WEST),
                 6: (Direction.WEST, Direction.WEST),
                 7: (Direction.NORTH, Direction.WEST)}
    return random.choice(list(wind_dirs.values()))

  def reset_wind(self):
    self.windspeed = self.set_windspeed()
    self.wind_dir = self.set_wind_dir()
    print("windspeed: ", self.windspeed)
    print(self.wind_dir)
    for node_row in self.nodes:
      for node in node_row:
        node.windspeed = self.windspeed
        node.wind_dir = self.wind_dir

  # Start agents at random positions
  def reset_agents(self):
    self.agents.clear()
    for agent in range(0, self.nr_of_agents):
      x, y = 0, 0
      while not (agentRadius < math.sqrt((x - self.size / 2)**2 + (y - self.size / 2)**2) < self.size / 2  and 0 < x < self.size and 0 < y < self.size):
        x, y = (random.randint(0, self.size), random.randint(0, self.size))
      self.agents += [Agent((x, y), self)]

    # if we have 30 agents the distribution should not matter that much
    #for agent in range(0, self.nr_of_agents, 4):
    #  agent_pos = self.get_random_position(1)
    #  while not self.position_in_bounds(agent_pos) or agent_pos in list(self.firepos):
    #    agent_pos = self.get_random_position(1)
    #  self.agents += [Agent(agent_pos, self)]
    #for agent in range(1, self.nr_of_agents, 4):
    #  agent_pos = self.get_random_position(2)
    #  while not self.position_in_bounds(agent_pos) or agent_pos in list(self.firepos):
    #    agent_pos = self.get_random_position(2)
    #  self.agents += [Agent(agent_pos, self)]
    #for agent in range(2, self.nr_of_agents, 4):
    #  agent_pos = self.get_random_position(3)
    #  while not self.position_in_bounds(agent_pos) or agent_pos in list(self.firepos):
    #    agent_pos = self.get_random_position(3)
    #  self.agents += [Agent(agent_pos, self)]
    #for agent in range(3, self.nr_of_agents, 4):
    #  agent_pos = self.get_random_position(4)
    #  while not self.position_in_bounds(agent_pos) or agent_pos in list(self.firepos):
    #    agent_pos = self.get_random_position(4)
    #  self.agents += [Agent(agent_pos, self)]


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
    # if self.time % 5 == 0:        # every 5 time steps new waypoints should be set
    #  print("agents require new waypoints")
    self.time += 1  # fire and agents need this info
    # 2
    if not self.agents:
      print("all agents dead")
      self.discard_episode()
      self.start_episode()
      return

    for agent in self.agents:
      if not self.find_node(agent.position).state == None and self.find_node(agent.position).state == NodeState.ON_FIRE:
        agent.dead = True
        print("agent dies")
        self.agents.remove(agent)
      agent.timestep(self.time)  # walks 1 step towards current waypoint & digs on the way

    for _ in range(fire_step_multiplicator):           # multiplicator of fire speed basically
      for node_row in self.nodes:
        for node in node_row:
          node.time_step()
      for node_row in self.nodes:
        for node in node_row:
          node.update_state()



    # 4
    # self.waypoints.clear() # Reset selection
    if self.state == State.FIRE_OUT_OF_CONTROL:
      # M do not safe the gathered data points of the episode
      self.start_episode()
      self.reset_necessary = True  # M view needs to be updated by controller... not a nice way but works
      return

    # 5
    # IF fire contained
    # self.DataSaver.append_episode()
    # self.start_episode()
    # return  # I guess it has to return to start new

    for subscriber in self.subscribers:
      subscriber.update(UpdateType.TIMESTEP_COMPLETE)

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

  def get_random_position(self, id):
    if id == 1:  ## top left
      x = random.randint(0, int(self.size / 2))
      y = random.randint(int(self.size / 2), (self.size - 1))
      while (x > (int(self.size / 2) - self.radius) and y < int(self.size / 2) + self.radius):  ##within radius
        x = random.randint(0, int(self.size / 2))
        y = random.randint(int(self.size / 2), (self.size - 1))
      return (x, y)

    elif id == 2:  ##top right
      x = random.randint(int(self.size / 2), (self.size - 1))
      y = random.randint(int(self.size / 2), (self.size - 1))
      while (x < (int(self.size / 2) + self.radius) and y < (
              int(self.size / 2) + self.radius)):  ##within centre radius
        x = random.randint(int(self.size / 2), (self.size - 1))
        y = random.randint(int(self.size / 2), (self.size - 1))
      return (x, y)

    elif id == 3:  ##bottom left
      x = random.randint(0, int(self.size / 2))
      y = random.randint(0, int(self.size / 2))
      while (x > (int(self.size / 2) - self.radius) and y > (
              int(self.size / 2) - self.radius)):  ##within centre radius
        x = random.randint(0, int(self.size / 2))
        y = random.randint(0, int(self.size / 2))
      return (x, y)

    elif id == 4:  ##bottom right
      x = random.randint(int(self.size / 2), self.size - 1)
      y = random.randint(0, int(self.size / 2))
      while (x < int(self.size / 2) + self.radius and y > int(
              self.size / 2) - self.radius):  ##within centre radius
        x = random.randint(int(self.size / 2), self.size - 1)
        y = random.randint(0, int(self.size / 2))
      return (x, y)
    print("error placing agent!")
    return

  def is_firebreak(self, position):
    return position in self.firebreaks

  def position_in_bounds(self, position):
    x, y = position
    return 0 <= x < self.size and 0 <= y < self.size

  ## Model manipulation
  def start_collecting_waypoints(self):
    print("wp: ", len(self.waypoints))
    self.waypoints.clear()
    self.waypoints_walking.clear()
    self.waypoints_digging.clear()
    print("wp: ", len(self.waypoints))

  def highlight_agent(self, agent_no):
    if agent_no is None:
      self.highlighted_agent = None
      for subscriber in self.subscribers:
        subscriber.update(UpdateType.HIGHLIGHT_AGENT)
      return

    self.highlighted_agent = self.agents[agent_no]
    for subscriber in self.subscribers:
      subscriber.update(UpdateType.HIGHLIGHT_AGENT, agent=self.highlighted_agent)
    
    #print(f"{self.highlighted_agent.angle()} for pos {self.highlighted_agent.position} - {self.highlighted_agent.pos_rel_to_centre()}")

  def undo_selection(self, agent_no):
    for subscriber in self.subscribers:
      subscriber.update(UpdateType.CLEAR_WAYPOINTS, agent=self.highlighted_agent)
    
    self.waypoints.clear()
    self.waypoints_digging.clear()
    self.waypoints_walking.clear()
    
    for agent in self.agents[0:agent_no]:
      if agent.waypoint is not None:
        self.waypoints.add(agent.position)              # keep old waypoints for now
        if agent.is_digging:
          self.waypoints_digging.add(agent.position)
        else:
          self.waypoints_walking.add(agent.position)
        for subscriber in self.subscribers:
          subscriber.update(UpdateType.WAYPOINT, position=agent.original_waypoint)

    self.highlight_agent(agent_no)

  def select_square(self, position, digging):
    if self.highlighted_agent == None:
      return

    self.highlighted_agent.assign_new_waypoint(position, digging)

    self.waypoints.add(position)              # keep old waypoints for now
    if digging:
      self.waypoints_digging.add(position)
    else:
      self.waypoints_walking.add(position)
    for subscriber in self.subscribers:
      subscriber.update(UpdateType.WAYPOINT, position=position)

    # self.waypoint_history.add((self.time, position, self.highlighted_agent.position))

  ## State changes
  ## Call from Node
  def node_state_change(self, node: Node):
    if node.state == NodeState.ON_FIRE:
      self.firepos.add(node.position)
    if node.state == NodeState.BURNED_OUT:
      self.firepos.remove(node.position)
    if node.state == NodeState.FIREBREAK:
      self.firebreaks.add(node.position)

    for subscriber in self.subscribers:
      subscriber.update(UpdateType.NODE, node=node)

  def agent_moves(self, agent):
    for subscriber in self.subscribers:
      subscriber.update(UpdateType.AGENT, agent=agent)

  def subscribe(self, subscriber):
    self.subscribers.append(subscriber)


  def sort_agents_by_angle(self):
    self.agents.sort(key=lambda x:x.angle())


  ## Proper shutdown
  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    self.DataSaver.save_training_run()  # M data points of all successful episodes until here saved
    self.start_episode()
