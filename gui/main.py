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


def main():
  model = Model(size, nr_of_agents, agentRadius)            # Initialize Environment
  view = View(model, block_size_in_pixels)                  # Start View

  architecture_variants = ["xy", "angle", "box"]            # our 3 individual network output variants

  NN_control = True                                         # False -> gather data, True -> test NN
                                                            # Initialize Controller with model and view, NN stuff
  controller = Controller(model, view, NN_control, architecture_variants[0])

  if NN_control:
    while True:
      controller.update_NN(pygame.event.wait())
  else:
    while True:
      controller.update(pygame.event.wait())                # Let the controller take over



if __name__=="__main__":
  main()