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
  def find(coming_from, going_to):
    if coming_from == going_to:
      return Direction.NONE

    delta_x = going_to[0] - coming_from[0]
    delta_y = going_to[1] - coming_from[1]
    if abs(delta_x) > abs(delta_y):
      return Direction.GO_WEST if delta_x < 0 else Direction.GO_EAST
    return Direction.GO_NORTH if delta_y < 0 else Direction.GO_SOUTH

  @staticmethod
  def is_opposite(first, second):
    to_set = {first.value, second.value}
    if to_set == {"N", "S"} or to_set == {"E", "W"}:
      return True

    return False