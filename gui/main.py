### Required packages
# Pygame: pip install pygame

import pygame

from Model.model import Model
from Model.agent import Agent
from View.view import View
from Controller.controller import Controller

import random

random.seed(0)

pygame.init()  ## Initialize Pygame 

def random_position(size):
  return (random.randint(0, size), random.randint(0, size))


def main():
  # Environment parameters
  size = 25
  nr_of_agents = 5
  block_size_in_pixels = int(900 / size)

  agents = [Agent(random_position(size)) for _ in range(nr_of_agents)]

  # Initialization
  environment = Model(size, agents)               ## Initialize Environment
  view = View(environment, block_size_in_pixels)  ## Start View
  controller = Controller(environment, view)      ## Initialize Controller with model and view

  # Run 
  while(True):
    controller.update(pygame.event.get())         ## Let the controller take over


if __name__=="__main__":
  main()