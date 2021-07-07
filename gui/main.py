#!/bin/python3

# numba
# Tensorflow 2.x (when testing the CNN)

from Model.model import Model
from Controller.controller import Controller
import sys
from Model import utils

def main():
  architecture_variants = ["xy", "angle", "box"]  # our 3 individual network output variants
  experiments = ["BASIC", "STOCHASTIC", "WINDONLY", "UNCERTAINONLY", "UNCERTAIN+WIND"]
  variant = architecture_variants[2]
  experiment = experiments[1]
  n_NN_to_test = 1
  n_runs_per_NN = 2

  if len(sys.argv) > 1 and int(sys.argv[1]) < len(sys.argv):
      variant = architecture_variants[int(sys.argv[1])]
  if len(sys.argv) > 2 and int(sys.argv[2]) < len(experiments):
      experiment = experiments[int(sys.argv[2])]
  print(f"variant: {variant}")
  print(f"experiment: {experiment}")

  NN_control = True                                         # False -> gather data, True -> test NN
                                                            # Initialize Controller with model and view, NN stuff

  utils.configure_globals(experiment)
  model = Model(utils.size, utils.nr_of_agents, utils.agentRadius)            # Initialize Environment

  if NN_control:
    for NN_nr in range(n_NN_to_test):
      controller = Controller(model, NN_control, variant, NN_nr, n_runs_per_NN)
      while True:
        controller.update_NN_no_gui()
        if controller.next_model:                           # new controller with next model
          break

  #else:
  #  while True:
  #    controller.update(pygame.event.wait())                # Let the controller take over


if __name__=="__main__":
  main()


"""
conditions to fail:
- agent in fire
- 15 waypoint assignments done
- agent waypoint outside of env

conditions to win:
- BFS+heuristics can't find a way out"""