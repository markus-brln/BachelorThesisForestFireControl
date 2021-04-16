from gui.Model.direction import Direction
import time

# GLOBALS (documented when saving)
size = 255                              # environment size
nr_of_agents = 5
windspeed = 1
wind_dir = (Direction.NORTH)
timeframe = 20                          # timeframe in between setting new waypoints
agentRadius = 50                        # agents spawn in this radius around the fire (it's a box, not a circle, for now, see model.get_random_position())
randseed = time.time()

# OTHER GLOBALS
n_wind_dirs = 8
n_wind_speed_levels = 5
block_size_in_pixels = int(880 / size)