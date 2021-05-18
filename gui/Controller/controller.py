import enum
from typing import Container
import pygame
import time
from Model.model import Model
from View.view import View
from Model.utils import *
from Model.direction import Direction

import os
from pygame.locals import *

from enum import Enum

import tensorflow as tf
import numpy as np


class Controller:
  def __init__(self, model: Model, view: View, NN_control = False):
    self.model = model
    self.view = view

    # Initialization
    self.mouse_button_pressed = False   ## Mouse button assumed not to be pressed initially
    self.collecting_waypoints = False
    self.agent_no = 0
    self.last_timestep_waypoint_collection = -1

    # NN INTEGRATION
    self.NN_control = NN_control
    self.nn = None
    if self.NN_control:
      self.nn = self.load_NN()
    self.digging_threshold = 0.5


  def update(self, event):
    if event.type == pygame.QUIT:
      # Exit the program
      self.shut_down(event)

    if self.collecting_waypoints:
      self.collect_waypoints(event)
      return

    if self.model.reset_necessary:        # M update view when resetting env (hacky way)
      self.view.update()
      self.model.reset_necessary = False

    #for event in pygame_events:
    elif event.type == pygame.MOUSEBUTTONDOWN:
      # Mouse stationary and mouse button pressed
      self.mouse_button_pressed = event.button
      #self.select(event)
    elif event.type == pygame.MOUSEBUTTONUP:
      # Mouse button released
      self.mouse_button_pressed = False
      self.last_clicked = (-1, 0)           ## Reset to allow clicking a square twice in a row
    elif event.type == pygame.KEYDOWN:
      # Keyboard button pressed
      self.key_press(event)


  def shut_down(self, event):
    self.model.shut_down()              ## Ensure proper shutdown of the model
    exit(0)                             ## Exit program


  def select(self, event):
    # Determine the block the mouse is covering
    position = self.view.pixel_belongs_to_block(event.pos)
    # Select or deselect that block
    # if self.mouse_button_pressed == 1: ## Left click
    self.model.select_square(position)
    # else:                              ## Right click
    # self.model.deselect_square(position)

  def start_collecting_waypoints(self):
    print("Start collecting waypoints")
    self.view.clear_waypoints([self.model.find_node(pos) for pos in self.model.waypoints])
    self.model.waypoints.clear()    # clear the actual waypoint positions after deleting them on the view!

    self.collecting_waypoints = True
    self.model.sort_agents_by_angle()
    self.agent_no = 0
    self.model.highlight_agent(0)


  def collect_waypoints(self, event):
    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_LEFT:
        if self.agent_no != 0:
          self.model.undo_selection(self.agent_no - 1)
          self.agent_no -= 1

    if event.type != pygame.MOUSEBUTTONDOWN:
      return

    position = self.view.pixel_belongs_to_block(event.pos)

    #print(event.button)
    if event.button == 1:       # left mouse button, digging waypoint
      self.model.select_square(position, digging=True)
    elif event.button == 3:     # right mouse button, walking waypoint
      self.model.select_square(position, digging=False)
    else:
      print("use left(digging) or right (walking) mouse button")
      return

    self.agent_no += 1
    if self.agent_no >= len(self.model.agents):
      self.collecting_waypoints = False
      self.model.highlight_agent(None)
      return

    self.model.highlight_agent(self.agent_no)


  def key_press(self, event):
    if event.key == pygame.K_ESCAPE:
      self.model.DataSaver.save_training_run()

    if event.key == pygame.K_SPACE:
      if self.model.time % timeframe == 0:
        if self.last_timestep_waypoint_collection != self.model.time:
          self.start_collecting_waypoints()
          self.last_timestep_waypoint_collection = self.model.time
        else:
          self.model.append_datapoint()   # only start after first 'timeframe' timesteps
          #start = time.time()
          for _ in range(timeframe):
            self.model.time_step()          ## Space to go to next timestep

          #print("time: ", time.time()-start)

    if event.key == pygame.K_RETURN:
      self.model.append_episode()
      self.model.start_episode()       ## Return / ENTER to go to next episode
      self.model.reset_wind()
      self.last_timestep_waypoint_collection = -1 # first get new waypoints when restarting episode

    if event.key == pygame.K_BACKSPACE:
      self.model.discard_episode()
      self.model.start_episode()  ## Return / ENTER to go to next episode
      self.model.reset_wind()
      self.last_timestep_waypoint_collection = -1


  ## ALL NN STUFF STARTS HERE
  def update_NN(self, event):
    if event.type == pygame.QUIT:
      # save data about how often fire was contained
      exit()

    #print("event type:", event.type)
    if self.collecting_waypoints and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
      print("predicting...")
      outputs = self.predict_NN()
      print("outputs: ", outputs)

      print("Assigning waypoints based on NN outputs")
      self.set_waypoints_NN(outputs)

      print("Model progresses")
      for _ in range(timeframe):
        self.model.time_step()          ## Space to go to next timestep
      return

    if self.model.reset_necessary:        # M update view when resetting env (hacky way)
      self.view.update()
      self.model.reset_necessary = False

    elif not self.collecting_waypoints and event.type == pygame.KEYDOWN:
      self.start_collecting_waypoints()


  def set_waypoints_NN(self, outputs):
    print("len outputs", len(outputs))
    for output in outputs:
      new_wp = (int(output[0] * 255), int(output[1] * 255))
      digging = output[2] > self.digging_threshold
      #new_wp = self.view.pixel_belongs_to_block(event.pos)
      print("pos: ", new_wp, "dig: ", digging)
      self.model.highlight_agent(self.agent_no)
      self.model.select_square(new_wp, digging=digging)
      self.agent_no += 1
      time.sleep(0.5)


    self.collecting_waypoints = False
    self.model.highlight_agent(None)


  def predict_NN(self):
    """Use the pre-loaded CNN to generate waypoints for the agents.
    INPUT:
    - on-the-fly updated 5-channel image of environment
    - concat vector (wind dir + wind speed + agent pos)

    OUTPUT:
    - n_agents * [x, y, 0-1 value for drive/dig]
    """
    if len(self.model.agents) != 5:
      print("Agent(s) must have died")
      exit()

    self.plot_env_img(self.model.array_np)

    X1 = 5 * [self.model.array_np]
    wind_info = list(self.model.wind_info_vector)
    agent_positions = [agent.position for agent in self.model.agents]
    concat_vector = list()
    for pos in agent_positions:
      concat_vector.append(wind_info + [pos[0] / 255, pos[1] / 255])

    print(concat_vector)
    concat_vector = np.asarray(concat_vector)

    print("predicting")
    output = self.nn.predict([X1, concat_vector])                        # outputs 16x16x3
    return output


  def plot_env_img(self, img):
    # translate the 5 channel input back to displayable images
    orig_img = np.zeros((256, 256))

    for y, row in enumerate(img):
      for x, cell in enumerate(row):
        for idx, item in enumerate(cell):
          #print(y, x, idx)
          if item == 1:
            orig_img[y][x] = idx


    from matplotlib import pyplot as plt
    plt.imshow(orig_img)
    plt.show()


  @staticmethod
  def load_NN(filename="saved_models/CNN"):
    """Load a Keras model from json file and weights (.h5). Same as in
    CNN/NNutils.py"""
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print("loading model " + filename)
    # load json and create model
    json_file = open('saved_models' + os.sep + filename + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights('saved_models' + os.sep + filename + ".h5")
    print("Loaded model from disk")

    return model




class NN_Controller:
  def __init__(self, filename, model: Model):
    self.nn = self.load_NN(filename)

    for l in self.nn.layers:
      l.trainable = False

    self.model = model


  def run(self, episodes, timesteps = 20):
    for _ in range(episodes):
      self.model.start_episode()
      while self.model.firepos != set() and len(self.model.agents) == 5: # While firepos not empty #TODO
        NN_output = self.predict()       # Get NN output
        print("waiting")
        time.sleep(5)
        self.steer_model(NN_output)      # use output to assign waypoints

        #print("waiting")
        #self.wait_keypress()              # prevent
        for _ in range(timesteps):
          #time.sleep(0.5) # So we can see what's going on. Disable when running.
          self.model.time_step()


  def steer_model(self, nn_output):
    digging_threshold = 0.5

    # Assign waypoints to the agents
    for agent, output in zip(self.model.agents, nn_output):
      print(output)
      self.model.select_square((255 * output[0], 255 * output[1]), digging=output[2] > digging_threshold)
      #agent.assign_new_waypoint((255 * output[0], 255 * output[1]), output[2] > digging_threshold)


  def load_NN(self, filename):
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print("loading model " + filename)
    # load json and create model
    json_file = open('saved_models' + os.sep + filename + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(model_json)
    # load weights into new model
    model.load_weights('saved_models' + os.sep + filename + ".h5")
    print("Loaded model from disk")

    return model


  def predict(self):
    X1 = 5 * [self.model.array_np]
    wind_info = list(self.model.wind_info_vector)
    agent_positions = [agent.position for agent in self.model.agents]
    concat_vector = list()
    for pos in agent_positions:
      concat_vector.append(wind_info + [pos[0] / 255, pos[1] / 255])

    print(concat_vector)
    concat_vector = np.asarray(concat_vector)

    print("predicting")
    output = self.nn.predict([X1, concat_vector])                        # outputs 16x16x3
    return output

  def get_wind_dir_idx(self):
    wind_dir = self.model.wind_dir
    """Order of wind directions:
       N, S, E, W, NE, NW, SE, SW"""
    if wind_dir == (Direction.NORTH, Direction.NORTH):
      return 0
    if wind_dir == (Direction.SOUTH, Direction.SOUTH):
      return 1
    if wind_dir == (Direction.EAST, Direction.EAST):
      return 2
    if wind_dir == (Direction.WEST, Direction.WEST):
      return 3
    if wind_dir == (Direction.NORTH, Direction.EAST):
      return 4
    if wind_dir == (Direction.NORTH, Direction.WEST):
      return 5
    if wind_dir == (Direction.SOUTH, Direction.EAST):
      return 6
    if wind_dir == (Direction.SOUTH, Direction.WEST):
      return 7

  def wait_keypress(self):
    input("hi")
    return
    while True:
      for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_f:
          return