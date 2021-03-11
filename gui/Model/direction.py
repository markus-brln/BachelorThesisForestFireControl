from enum import Enum

class Direction(Enum):
  NONE = lambda pos: pos
  UP = lambda pos: (pos[0], pos[1] - 1)
  RIGHT = lambda pos: (pos[0] + 1, pos[1])
  DOWN = lambda pos: (pos[0], pos[1] + 1)
  LEFT = lambda pos: (pos[0] - 1, pos[1])

  @staticmethod
  def find(coming_from, going_to):
    if coming_from == going_to:
      return Direction.NONE

    delta_x = going_to[0] - coming_from[0]
    delta_y = going_to[1] - coming_from[1]
    if abs(delta_x) > abs(delta_y):
      return Direction.LEFT if delta_x < 0 else Direction.RIGHT
    return Direction.UP if delta_y < 0 else Direction.DOWN