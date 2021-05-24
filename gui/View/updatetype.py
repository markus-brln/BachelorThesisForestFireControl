from enum import Enum

class UpdateType(Enum):
  """Determines which kinds of nodes should be updated by the View,
  reduces computation of the gui."""
  RESET = 0
  NODE = 1
  AGENT = 2
  WAYPOINT = 3
  TIMESTEP_COMPLETE = 4
  HIGHLIGHT_AGENT = 5
  CLEAR_WAYPOINTS = 6
