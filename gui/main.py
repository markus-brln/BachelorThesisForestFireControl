#!/bin/python3

### Required packages
# Pygame: pip install pygame

import pygame
from Model.model import Model
from View.view import View
from Controller.controller import Controller
from Model.utils import *
pygame.init()  ## Initialize Pygame
pygame.display.set_caption('Only you can prevent Forest Fires!')

random.seed(0)

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
  environment = Model(size, nr_of_agents, agentRadius)   ## Initialize Environment
  view = View(environment, block_size_in_pixels)  ## Start View
  controller = Controller(environment, view)      ## Initialize Controller with model and view

  # Run 
  while True:
    controller.update(pygame.event.wait())        ## Let the controller take over


if __name__=="__main__":
  main()