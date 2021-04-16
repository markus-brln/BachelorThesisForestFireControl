from gui.Model.direction import Direction
from enum import IntEnum
import time
import random

random.seed(0)


# WIND values
windspeed = 0
wind_dir = []
n_wind_dirs = 7
n_wind_speed_levels = 5

# GLOBALS (documented when saving)
size = 255  # environment size
nr_of_agents = 12
timeframe = 20  # timeframe in between setting new waypoints
agentRadius = 30  # agents spawn in this radius around the fire (it's a box, not a circle, for now, see model.get_random_position())
randseed = time.time()

# OTHER GLOBALS
block_size_in_pixels = int(880 / size)
