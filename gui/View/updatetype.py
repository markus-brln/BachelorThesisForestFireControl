from enum import Enum

class UpdateType(Enum):
  RESET = 0
  NODE = 1
  AGENT = 2
  WAYPOINT = 3
  TIMESTEP_COMPLETE = 4
  HIGHLIGHT_AGENT = 5
  CLEAR_WAYPOINTS = 6
