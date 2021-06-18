import time
import math

# WIND values
n_wind_dirs = 8
n_wind_speed_levels = 5
wind_on = False

# GLOBALS (documented when saving)
size = 255                                                  # environment size
nr_of_agents = 5
timeframe = 20                                              # timeframe in between setting new waypoints
agentRadius = 70                                            # agents spawn in this radius around the fire
randseed = time.time()
fire_step_multiplicator = 0.3                                 # speed up / slow down fire
uncertain_spawn = 0

# OTHER GLOBALS
block_size_in_pixels = int(765 / size)
digging_threshold = 0.5                                     # NN output > threshold -> agent will dig (new architecture)
apd = 10                                                    # agent_point_diameter when constructing the CNN input image

def configure_globals(experiment):
    global wind_on
    global uncertain_spawn
    global fire_step_multiplicator

    if experiment == "BASIC":
        wind_on = False
        uncertain_spawn = 0
    elif experiment == "STOCHASTIC":
        wind_on = False
        uncertain_spawn = 10
    elif experiment == "WIND":
        wind_on = True
        uncertain_spawn = 10
        fire_step_multiplicator = 0.6

    elif experiment == "UNCERTAINTY":
        pass
    elif experiment == "UNCERTAINTY+WIND":
        pass


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) - math.cos(angle) * (py - oy)

    return int(qx), int(qy)