#!/bin/python3

### Required packages
# Pygame > 2.0.1: pip install pygame

# Tensorflow 2.x (when testing the CNN)

import pygame
from Model.model import Model
from View.view import View
from Controller.controller import Controller
from Model.utils import *

pygame.init()                                               # Initialize Pygame
pygame.display.set_caption('Only you can prevent Forest Fires!')

# def set_wind():
#   wind_dirs = {0: (Direction.NORTH, Direction.NORTH),
#                1: (Direction.NORTH, Direction.EAST),
#                2: (Direction.EAST, Direction.EAST),
#                3: (Direction.SOUTH, Direction.EAST),
#                4: (Direction.SOUTH, Direction.SOUTH),
#                5: (Direction.SOUTH, Direction.WEST),
#                6: (Direction.WEST, Direction.WEST),
#                7: (Direction.NORTH, Direction.WEST)}
#   windspeed = random.randint(0, n_wind_speed_levels)
#   wind_dir = random.choice(list(wind_dirs.values()))
#   print("windspeed: ", windspeed)
#   print(wind_dir)

def main():
  # Initialization
  model = Model(size, nr_of_agents, agentRadius)            # Initialize Environment
  view = View(model, block_size_in_pixels)                  # Start View

  NN_control = False                                         # False -> gather data, True -> test NN
  controller = Controller(model, view, NN_control)          # Initialize Controller with model and view

  if NN_control:
    while True:
      controller.update_NN(pygame.event.wait())
  else:
    while True:
      controller.update(pygame.event.wait())                # Let the controller take over



if __name__=="__main__":
  main()