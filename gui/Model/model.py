from Model.agent import Agent
from Model.direction import Direction
from Model.data_saver import DataSaver
from Model.node import Node, NodeType, NodeState
from View.updatetype import UpdateType
from Model import utils

from enum import Enum
import random
import math
import numpy as np

class ModelState(Enum):
  """Possible states of the model, will reset when fire out of control
     (outside borders of environment)."""
  ONGOING = 0
  FIRE_CONTAINED = 1
  FIRE_OUT_OF_CONTROL = 2


class Model:
  def __init__(self, length: int, n_agents: int, radius: int):
    self.counter = 0
    self.firebreaks = set()
    self.waypoints = set()
    self.wp_driving = set()
    self.wp_digging = set()
    self.time = 0
    self.subscribers = []
    self.size = length
    self.centre = (int(length / 2), int(length / 2))
    self.n_agents = n_agents
    self.agents = []
    self.state = ModelState.ONGOING
    self.reset_necessary = False
    self.wind_dir = self.set_wind_dir()
    self.wind_speed = self.set_windspeed()
    print("wind direction: ", self.wind_dir)
    print("wind speed: ", self.wind_speed)
    self.nodes = []
    self.init_nodes()

    # Fire initialization
    self.agent_radius = radius                              # distance of spawning of agents from the centre
    self.firepos = set()

    # Data saving initialization
    self.DataSaver = DataSaver(self)                        # receives reference to the model
    self.highlighted_agent_nr = None
    self.highlighted_agent = None

    # NN integration
    shape = (256, 256, 5)
    self.array_np = np.zeros(shape, dtype=np.double)
    self.wind_info_vector = np.zeros(13, dtype=np.double)   # 8 wind directions
    self.start_episode()


  def start_episode(self):
    self.counter += 1
    print(f"{self.counter}th run")
    self.reset_agents()
    self.waypoints = set()
    self.wp_driving = set()
    self.wp_digging = set()
    self.time = 0
    self.state = ModelState.ONGOING

    for node_row in self.nodes:
      for node in node_row:
        node.reset()

    # Start fire in the middle of the map
    self.firepos.clear()
    self.set_initial_fire()
    self.firebreaks = set()

    # NN integration
    shape = (256, 256, 5)
    self.array_np = np.zeros(shape, dtype=np.double)        # reset the NN's picture of the env
    for rowIdx, row in enumerate(self.array_np):
      for colIdx, _ in enumerate(row):
        self.array_np[rowIdx][colIdx][0] = 1


    for agent in self.agents:
      x, y = agent.position
      self.array_np[x][y][0] = 0
      self.array_np[x][y][4] = 1
    
    self.wind_info_vector = np.zeros(utils.n_wind_dirs + utils.n_wind_speed_levels, dtype=np.uint8)
    self.wind_info_vector[self.get_wind_dir_idx()] = 1
    self.wind_info_vector[self.wind_speed + 8] = 1           # +8 wind directions

    for subscriber in self.subscribers:
      subscriber.update(UpdateType.RESET)

  def init_nodes(self):
    """Construct Nodes, set their neighbours to speed up retrieval of that
       information."""
    # Set all Nodes to be forest
    for x in range(self.size):
      node_row = []
      for y in range(self.size):
        newNode = Node(self, (x, y), NodeType.TREES, self.wind_dir, self.wind_speed)
        node_row.append(newNode)
      self.nodes.append(node_row)

    # Set the neighbours
    for node_row in self.nodes:
      for node in node_row:
        pos = node.position
        neighbour = {"N": Direction.GO_NORTH(pos), "E": Direction.GO_EAST(pos),  ## Neighbours
                     "S": Direction.GO_SOUTH(pos), "W": Direction.GO_WEST(pos)}

        node.set_neighbours(north=self.find_node(neighbour["N"]),
                            east=self.find_node(neighbour["E"]),
                            south=self.find_node(neighbour["S"]),
                            west=self.find_node(neighbour["W"]))


  def set_initial_fire(self):
    """Ignite the centre node."""
    x = y = int(self.size / 2)
    centre_node = self.find_node((x, y))
    centre_node.ignite()
    self.firepos.add((x, y))


  def reset_wind(self):
    """Reset values for wind speed and direction held by model and
       each individual node."""
    self.wind_speed = self.set_windspeed()

    self.wind_dir = self.set_wind_dir()
    print("windspeed: ", self.wind_speed)
    print(self.wind_dir)
    for node_row in self.nodes:
      for node in node_row:
        node.wind_speed = self.wind_speed
        node.wind_dir = self.wind_dir


  def reset_agents(self):
    """Place agents at random positions in a donut shape around the fire."""
    self.agents.clear()
    angle = 0
    orig_point = self.centre[0] - self.agent_radius, self.centre[1]
    for agent in range(0, self.n_agents):
      spawn_point = utils.rotate_point(self.centre, orig_point, angle)
      spawn_point = spawn_point[0] + random.randint(-utils.uncertain_spawn, utils.uncertain_spawn), \
                    spawn_point[1] + random.randint(-utils.uncertain_spawn, utils.uncertain_spawn)
      self.agents += [Agent(spawn_point, self)]
      angle += math.pi * 2 / self.n_agents              # star formation around centre


    # HARD WAY of setting agent positions
    '''for agent in range(0, self.nr_of_agents):
      x, y = 0, 0
      while not (agentRadius < math.sqrt((x - self.size / 2)**2 + (y - self.size / 2)**2) < self.size / 2  and 0 < x < self.size and 0 < y < self.size):
        x, y = (random.randint(0, self.size), random.randint(0, self.size))
      self.agents += [Agent((x, y), self)]'''


  def time_step(self):
    """This is executed 'timeframe' times in between waypoint assignments.
    - make agents move, remove if dead
    - node time steps (fire expansion)
    - update view
    """
    self.time += 1                                          # fire and agents need this info
    if not self.agents:
      print("all agents dead")
      self.discard_episode()
      self.start_episode()
      return

    for agent in self.agents:
      if not self.find_node(agent.position).state is None \
              and self.find_node(agent.position).state == NodeState.ON_FIRE:
        agent.dead = True
        print("agent dies")
        self.agents.remove(agent)
      agent.timestep()                                      # walks 1 step towards current waypoint & digs on the way


    if utils.fire_step_multiplicator >= 1:
      for _ in range(utils.fire_step_multiplicator):              # multiplicator of fire speed basically
        for node_row in self.nodes:
          for node in node_row:
            node.time_step()
    elif 0 < utils.fire_step_multiplicator < 1:
      if random.uniform(0, 1) < utils.fire_step_multiplicator:  # slow fire down below 1 step / timestep for easy envs
        for node_row in self.nodes:
          for node in node_row:
            node.time_step()
    else:
      print("invalid fire_step_multiplicator! exiting")
      exit()

    for node_row in self.nodes:  # update states anyway for gui to catch new firebreaks
      for node in node_row:
        node.update_state()

    if self.state == ModelState.FIRE_OUT_OF_CONTROL:             # do not safe the gathered data points of the episode
      self.start_episode()
      self.reset_necessary = True                           # view needs to be updated by controller... not a nice way but works
      return

    for subscriber in self.subscribers:
      subscriber.update(UpdateType.TIMESTEP_COMPLETE)


  def find_node(self, pos):
    if not self.position_in_bounds(pos):
      return None
    return self.nodes[pos[0]][pos[1]]


  def get_neighbours(self):
    """NOT USED"""
    return {"N": self.find_node(Direction.GO_NORTH),
            "E": self.find_node(Direction.GO_EAST),
            "S": self.find_node(Direction.GO_SOUTH),
            "W": self.find_node(Direction.GO_WEST)}


  def get_random_position(self, ID):
    """NOT USED"""
    if ID == 1:  ## top left
      x = random.randint(0, int(self.size / 2))
      y = random.randint(int(self.size / 2), (self.size - 1))
      while x > (int(self.size / 2) - self.agent_radius) and y < int(self.size / 2) + self.agent_radius:  ##within radius
        x = random.randint(0, int(self.size / 2))
        y = random.randint(int(self.size / 2), (self.size - 1))
      return x, y

    elif ID == 2:  ##top right
      x = random.randint(int(self.size / 2), (self.size - 1))
      y = random.randint(int(self.size / 2), (self.size - 1))
      while (x < (int(self.size / 2) + self.agent_radius) and y < (
              int(self.size / 2) + self.agent_radius)):  ##within centre radius
        x = random.randint(int(self.size / 2), (self.size - 1))
        y = random.randint(int(self.size / 2), (self.size - 1))
      return x, y

    elif ID == 3:  ##bottom left
      x = random.randint(0, int(self.size / 2))
      y = random.randint(0, int(self.size / 2))
      while (x > (int(self.size / 2) - self.agent_radius) and y > (
              int(self.size / 2) - self.agent_radius)):  ##within centre radius
        x = random.randint(0, int(self.size / 2))
        y = random.randint(0, int(self.size / 2))
      return x, y

    elif ID == 4:  ##bottom right
      x = random.randint(int(self.size / 2), self.size - 1)
      y = random.randint(0, int(self.size / 2))
      while (x < int(self.size / 2) + self.agent_radius and y > int(
              self.size / 2) - self.agent_radius):  ##within centre radius
        x = random.randint(int(self.size / 2), self.size - 1)
        y = random.randint(0, int(self.size / 2))
      return x, y
    print("error placing agent!")
    return


  def position_in_bounds(self, position):
    x, y = position
    return 0 <= x < self.size and 0 <= y < self.size


  def start_collecting_waypoints(self):
    """NOT USED"""
    print("wp: ", len(self.waypoints))
    self.waypoints.clear()
    self.wp_driving.clear()
    self.wp_digging.clear()
    print("wp: ", len(self.waypoints))


  def highlight_agent(self, agent_no):
    """Highlight agent yellow that is about to get a new waypoint."""
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
    """Delete last waypoint in order to redo it."""
    for subscriber in self.subscribers:
      subscriber.update(UpdateType.CLEAR_WAYPOINTS, position=[self.find_node(pos) for pos in self.waypoints])
    
    self.waypoints.clear()
    self.wp_digging.clear()
    self.wp_driving.clear()
    
    for agent in self.agents[:agent_no]:
      if agent.wp is not None:
        self.waypoints.add(agent.original_waypoint)
        if agent.is_digging:
          self.wp_digging.add(agent.original_waypoint)
        else:
          self.wp_driving.add(agent.original_waypoint)
        for subscriber in self.subscribers:
          subscriber.update(UpdateType.WAYPOINT, position=agent.original_waypoint)

    print(self.waypoints)
    self.highlight_agent(agent_no)


  def select_square(self, position, digging):
    if self.highlighted_agent is None:
      return

    self.highlighted_agent.assign_new_waypoint(position, digging)

    self.waypoints.add(position)
    if digging:
      self.wp_digging.add(position)
    else:
      self.wp_driving.add(position)
    for subscriber in self.subscribers:
      subscriber.update(UpdateType.WAYPOINT, position=position)

    # self.waypoint_history.add((self.time, position, self.highlighted_agent.position))

  ## State changes
  ## Call from Node
  def node_state_change(self, node: Node):
    """Node informs the model that it's state changed, so the
       internal representation (env matrix and fire/agent lists)
       is adjusted."""
    x, y = node.position
    self.array_np[x][y][0] = 0                              # No longer grass
    if node.state == NodeState.ON_FIRE:
      self.array_np[x][y][2] = 1
      self.firepos.add(node.position)
    if node.state == NodeState.BURNED_OUT:
      self.array_np[x][y][3] = 1
      self.firepos.remove(node.position)
    if node.state == NodeState.FIREBREAK:
      self.array_np[x][y][1] = 1
      self.firebreaks.add(node.position)

    for subscriber in self.subscribers:
      subscriber.update(UpdateType.NODE, node=node)


  def agent_moves(self, agent):
    old_x, old_y = agent.prev_node.position
    new_x, new_y = agent.position

    cell = 0                                                # agent.prev_node.state == NodeState.NORMAL
    if agent.prev_node.state == NodeState.FIREBREAK:
      cell = 1
    elif agent.prev_node.state == NodeState.ON_FIRE:
      cell = 2
    elif agent.prev_node.state == NodeState.BURNED_OUT:
      cell = 3

    self.array_np[old_x][old_y][cell] = 1
    self.array_np[old_x][old_y][4] = 0

    if agent.node.state == NodeState.NORMAL:
      cell = 0
    elif agent.node.state == NodeState.FIREBREAK:
      cell = 1
    elif agent.node.state == NodeState.ON_FIRE:
      cell = 2
    elif agent.node.state == NodeState.BURNED_OUT:
      cell = 3
    self.array_np[new_x][new_y][cell] = 0
    self.array_np[new_x][new_y][4] = 1

    for subscriber in self.subscribers:
      subscriber.update(UpdateType.AGENT, agent=agent)


  @staticmethod
  def set_windspeed():
    """Values from 0 to including 4 means there are 5 wind speed levels"""
    if utils.wind_on:
      return random.randint(0, utils.n_wind_speed_levels-1)
    else:
      return 0

  @staticmethod
  def set_wind_dir():
    """Choose randomly out of 8 possible wind directions."""
    wind_dirs = {0: (Direction.NORTH, Direction.NORTH),
                 1: (Direction.NORTH, Direction.EAST),
                 2: (Direction.EAST, Direction.EAST),
                 3: (Direction.SOUTH, Direction.EAST),
                 4: (Direction.SOUTH, Direction.SOUTH),
                 5: (Direction.SOUTH, Direction.WEST),
                 6: (Direction.WEST, Direction.WEST),
                 7: (Direction.NORTH, Direction.WEST)}
    if utils.wind_on:
      return random.choice(list(wind_dirs.values()))
    else:
      return list(wind_dirs.values())[0]


  def subscribe(self, subscriber):
    """View becomes subscriber to changes in the model."""
    self.subscribers.append(subscriber)


  def sort_agents_by_angle(self):
    """Makes waypoint assgignment start on the left above the middle in
    clock-wise fashion to achieve faster data gathering."""
    self.agents.sort(key=lambda x:x.angle())


  def append_datapoint(self):
    print("Saving waypoints")
    self.DataSaver.append_datapoint()

  def output_datapoints(self):
    print("testing data pass")
    return self.DataSaver.output_datapoint()

  def discard_episode(self):
    self.DataSaver.discard_episode()

  def append_episode(self):
    self.DataSaver.append_episode()

  def save_training_run(self):
    self.DataSaver.save_training_run()

  def count_containment(self):
    """Important for testing, counts the amount of potentially
    burned cells when fire was contained."""
    burned = [(int(self.size/2), int(self.size/2))]         # starting from the middle like the fire
    new_burned = burned.copy()
    previous_n = 0                                          # previous amount of potentially burned cells

    while len(burned) > previous_n:
      previous_n = len(burned)
      new_new_burned = []
      for pos in new_burned:
        neighbours = [(pos[0] + 1, pos[1]),(pos[0], pos[1] + 1),(pos[0] - 1, pos[1]),(pos[0], pos[1] - 1)]
        for neighbour in neighbours:
          if neighbour not in self.firebreaks and neighbour not in burned and neighbour not in new_new_burned:
            new_new_burned.append(neighbour)
      new_burned = new_new_burned
      burned.extend(new_burned)

    return len(burned)


  def shut_down(self):
    """Saves the data gathering of the current run, initializes new run (episode)."""
    self.DataSaver.save_training_run()                      # data points of all successful episodes until here saved
    self.start_episode()


  def get_wind_dir_idx(self):
    """Save wind direction index in wind dir vector depending on the combined
    directions present in the model.

    Order of wind directions:
           N, S, E, W, NE, NW, SE, SW"""
    wind_dir = self.wind_dir
    if wind_dir == (Direction.NORTH, Direction.NORTH):
      return 0
    if wind_dir == (Direction.SOUTH, Direction.SOUTH):
      return 1
    if wind_dir == (Direction.EAST, Direction.EAST):
      return 2
    if wind_dir == (Direction.WEST, Direction.WEST):
      return 3
    if wind_dir == (Direction.NORTH, Direction.EAST):
      return 4
    if wind_dir == (Direction.NORTH, Direction.WEST):
      return 5
    if wind_dir == (Direction.SOUTH, Direction.EAST):
      return 6
    if wind_dir == (Direction.SOUTH, Direction.WEST):
      return 7