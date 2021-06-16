#!/bin/python3

### Required packages
# Pygame > 2.0.1: pip install pygame

# Tensorflow 2.x (when testing the CNN)

import pygame
from Model.model import Model
from View.view import View
from Controller.controller import Controller
from Model.utils import *

import sys

pygame.init()                                               # Initialize Pygame
pygame.display.set_caption('Only you can prevent Forest Fires!')


def main():
	
  model = Model(size, nr_of_agents, agentRadius)            # Initialize Environment
  view = View(model, block_size_in_pixels)                  # Start View

  architecture_variants = ["xy", "angle", "box"]            # our 3 individual network output variants
  if len(sys.argv) > 1 and int(sys.argv[1]) < len(sys.argv):
    variant = architecture_variants[int(sys.argv[1])]
  else:
    variant = architecture_variants[2]
   
  print(f"variant: {variant}")
  

  NN_control = False                                         # False -> gather data, True -> test NN
                                                            # Initialize Controller with model and view, NN stuff
  controller = Controller(model, view, NN_control, variant)

  if NN_control:
    while True:
      controller.update_NN(pygame.event.wait())
  else:
    while True:
      controller.update(pygame.event.wait())                # Let the controller take over



if __name__=="__main__":
  main()


# TODO NOTES
'''
- PEREGRINE!!!
- generate data in a certain way (show), move to the outside once finished
- make basic env work for all architectures
- including current agent position in input works better than concatenating

- show where to integrate other outputs

4 different environments to test:
- basic 
- normal fire speed               # make each of us responsible for collecting data for one of 
                                  # those! so that our different strategies won't interfere
- normal + wind dir + wind speed

additional experiments (for each env variant):
- 3 different levels of amounts of training data

'''