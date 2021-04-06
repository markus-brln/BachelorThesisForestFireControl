#!/bin/python3

### Required packages
# Pygame: pip install pygame

import pygame

from Model.environment import Model
from Model.direction import Direction
from Model.agent import Agent
from View.view import View
from Controller.controller import Controller


pygame.init()  ## Initialize Pygame
pygame.display.set_caption('Only you can prevent Forest Fires!')

def main():
  # Environment parameters
  size = 99
  nr_of_agents = 7
  agentRadius = 25
  block_size_in_pixels = int(880 / size)


  # Initialization
  environment = Model(size, nr_of_agents, agentRadius, wind_dir=Direction.EAST)   ## Initialize Environment
  view = View(environment, block_size_in_pixels)  ## Start View
  controller = Controller(environment, view)      ## Initialize Controller with model and view

  # Run 
  while True:
    controller.update(pygame.event.wait())        ## Let the controller take over


if __name__=="__main__":
  main()