import time

# WIND values
windspeed = 0
wind_dir = []
n_wind_dirs = 8
n_wind_speed_levels = 5

# GLOBALS (documented when saving)
size = 255                                                  # environment size
nr_of_agents = 5
timeframe = 20                                              # timeframe in between setting new waypoints
agentRadius = 70                                            # agents spawn in this radius around the fire
randseed = time.time()
fire_step_multiplicator = 1                                 # int!! to speed up fire

# OTHER GLOBALS
block_size_in_pixels = int(765 / size)
