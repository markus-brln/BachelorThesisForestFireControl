### Required packages
# Pygame: pip install pygame

import pygame

from Model.model import Model
from Model.agent import Agent
from View.view import View
from Controller.controller import Controller


pygame.init()  ## Initialize Pygame
pygame.display.set_caption('Only you can prevent Forest Fires!')

def main():
  # Environment parameters
  size = 25
  nr_of_agents = 5
  firesize = 1
  block_size_in_pixels = int(900 / size)


  # Initialization
  environment = Model(size, nr_of_agents, firesize)               ## Initialize Environment
  view = View(environment, block_size_in_pixels)  ## Start View
  controller = Controller(environment, view)      ## Initialize Controller with model and view

  # Run 
  while(True):
    controller.update(pygame.event.get())         ## Let the controller take over


if __name__=="__main__":
  main()