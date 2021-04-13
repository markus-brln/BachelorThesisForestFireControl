from Model.direction import Direction
import time

# GLOBALS
size = 251                              # environment size
nr_of_agents = 10
agentRadius = 50                        # agents spawn in this radius around the fire (it's a box, not a circle, for now, see model.get_random_position())
block_size_in_pixels = int(880 / size)
windspeed = 0
wind_dir = Direction.EAST
randseed = time.time()
timeframe = 50                          # timeframe in between setting new waypoints

