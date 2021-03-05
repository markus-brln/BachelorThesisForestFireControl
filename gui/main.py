### Required packages
# Pygame: pip install pygame

import pygame

from Model.model import Model
from View.view import View
from Controller.controller import Controller


pygame.init()  ## Initialize Pygame 

def main():
  # Environment parameters
  size = 25
  block_size_in_pixels = int(900 / size)

  # Initialization
  environment = Model(size, [])                   ## Initialize Environment
  view = View(environment, block_size_in_pixels)  ## Start View
  controller = Controller(environment, view)      ## Initialize Controller with model and view

  # Run 
  while(True):
    controller.update(pygame.event.get())         ## Let the controller take over


if __name__=="__main__":
  main()