#!/bin/python3

### Required packages
# Pygame > 2.0.1: pip install pygame

# Tensorflow 2.x (when testing the CNN)

import pygame
from Model.model import Model
from View.view import View
from Controller.controller import Controller
import sys
from Model import utils

pygame.init()                                               # Initialize Pygame
pygame.display.set_caption('Only you can prevent Forest Fires!')


def main():
  NN_control = True                                         # False -> gather data, True -> test NN

  architecture_variants = ["xy", "angle", "box", "segments"]  # our 3 individual network output variants
  experiments = ["BASIC", "STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
  variant = architecture_variants[0]
  experiment = experiments[1]
  NN_number = 0

  if len(sys.argv) > 1 and int(sys.argv[1]) < len(architecture_variants):
      variant = architecture_variants[int(sys.argv[1])]
  if len(sys.argv) > 2 and int(sys.argv[2]) < len(experiments):
      experiment = experiments[int(sys.argv[2])]
  if len(sys.argv) > 3 and int(sys.argv[3]) < 10:
    NN_number = int(sys.argv[3])

  print(f"variant: {variant}")
  print(f"experiment: {experiment}")

  utils.configure_globals(experiment)
  model = Model(utils.size, utils.nr_of_agents, utils.agentRadius)            # Initialize Environment
  view = View(model, utils.block_size_in_pixels)                  # Start View
  controller = Controller(model, view, NN_control, variant, NN_number)

  if NN_control:
    while True:
      controller.update_NN(pygame.event.wait())
  else:
    while True:
      controller.update(pygame.event.wait())                # Let the controller take over



if __name__=="__main__":
  main()


"""
conditions to fail:
- agent in fire
- 20 waypoint assignments done
- agent waypoint outside of env
conditions to win:
- BFS+heuristics can't find a way out"""