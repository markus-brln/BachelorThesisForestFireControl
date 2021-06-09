import time
import math

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
fire_step_multiplicator = 0.3                                 # speed up / slow down fire

# OTHER GLOBALS
block_size_in_pixels = int(765 / size)
digging_threshold = 0.5                                     # NN output > threshold -> agent will dig (new architecture)


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) - math.cos(angle) * (py - oy)

    return int(qx), int(qy)