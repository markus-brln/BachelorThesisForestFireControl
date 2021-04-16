from enum import Enum


class Direction(Enum):
  NONE = None
  NORTH = "N"
  SOUTH = "S"
  EAST = "E"
  WEST = "W"

  GO_NONE = lambda pos: pos
  GO_NORTH = lambda pos: (pos[0], pos[1] - 1)
  GO_EAST = lambda pos: (pos[0] + 1, pos[1])
  GO_SOUTH = lambda pos: (pos[0], pos[1] + 1)
  GO_WEST = lambda pos: (pos[0] - 1, pos[1])

  @staticmethod
  def dist2D(pos1, pos2):
    return pow(pow(pos1[0]-pos2[0], 2) + pow(pos1[1]-pos2[1], 2), 0.5) # pow(,0.5) == sqrt()


  @staticmethod
  def find(agent):
    """Line drawing algorithm to move an agent closer to its goal.
       Basically evaluates a linear equation for a suitable x or y
       (depending on incline) and then finds the best step direction
       to approximate that line."""
    if agent.position == agent.waypoint:    # agent will stay at its position when waypoint too close
      return Direction.GO_NONE

    delta_x = agent.waypoint[0] - agent.start_pos[0] # get a linear function of agent's start -> waypoint
    delta_y = agent.waypoint[1] - agent.start_pos[1]

    evaluate_at_x = False
    if abs(delta_x) >= abs(delta_y):        # move along straight line by adding 1 to x
      evaluate_at_x = True                  # less than 45 degree incline

    line_point = list()
    if evaluate_at_x and delta_x != 0:
      x = agent.position[0] + delta_x / abs(delta_x) # move left or right
      y = agent.start_pos[1] + (x - agent.start_pos[0]) * (delta_y / delta_x)
      line_point = [x, y]
    elif delta_y != 0:
      y = agent.position[1] + delta_y / abs(delta_y)  # move up or down
      x = agent.start_pos[0] + (y - agent.start_pos[1]) * (delta_x / delta_y)
      line_point = [x, y]
    if delta_x == 0 and delta_y == 0:
      line_point = [agent.position[0], agent.position[1]]

    # try to find the direction that leads to the closest approximation of the line
    distances = list()
    distances.append(Direction.dist2D([agent.position[0], agent.position[1] - 1], line_point)) # N
    distances.append(Direction.dist2D([agent.position[0], agent.position[1] + 1], line_point)) # S
    distances.append(Direction.dist2D([agent.position[0] + 1, agent.position[1]], line_point)) # E
    distances.append(Direction.dist2D([agent.position[0] - 1, agent.position[1]], line_point)) # W

    best_action_idx = distances.index(min(distances))
    if best_action_idx == 0:
      return Direction.GO_NORTH
    if best_action_idx == 1:
      return Direction.GO_SOUTH
    if best_action_idx == 2:
      return Direction.GO_EAST
    if best_action_idx == 3:
      return Direction.GO_WEST

    # the old way:
    '''delta_x = going_to[0] - agent.position[0]
    delta_y = going_to[1] - agent.position[1]
    if abs(delta_x) > abs(delta_y):
      return Direction.GO_WEST if delta_x < 0 else Direction.GO_EAST
    return Direction.GO_NORTH if delta_y < 0 else Direction.GO_SOUTH'''

  @staticmethod
  def is_opposite(first, second):
    to_set = {first.value, second.value}
    if to_set == {"N", "S"} or to_set == {"E", "W"}:
      return True

    return False