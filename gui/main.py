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

def main():
  # Initialization
  environment = Model(size, nr_of_agents, agentRadius, windspeed, wind_dir)   ## Initialize Environment
  view = View(environment, block_size_in_pixels)  ## Start View
  controller = Controller(environment, view)      ## Initialize Controller with model and view

  # Run 
  while True:
    controller.update(pygame.event.wait())        ## Let the controller take over


if __name__=="__main__":
  main()