import pygame
import time
from gui.Model.model import Model
from gui.View.view import View
from gui.Model.utils import *

class Controller:
  def __init__(self, model: Model, view: View):
    self.model = model
    self.view = view

    # Initialization
    self.mouse_button_pressed = False   ## Mouse button assumed not to be pressed initially
    self.collecting_waypoints = False
    self.agent_no = 0
    self.last_timestep_waypoint_collection = -1

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
      self.select(event)
    elif event.type == pygame.MOUSEBUTTONUP:
      # Mouse button released
      self.mouse_button_pressed = False
      self.last_clicked = (-1, 0)           ## Reset to allow clicking a square twice in a row
    # M dragging not useful for now
    #elif event.type == pygame.MOUSEMOTION and self.mouse_button_pressed:
      # Mouse button pressed and dragged
      #self.select(event)
    # M having this in here makes it extremely laggy, at least for me
    #elif event.type == pygame.MOUSEMOTION: ## now will constantly display mouse coords
      #self.view.update()
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
    print("Assigning waypoints")
    self.view.clear_waypoints([self.model.find_node(pos) for pos in self.model.waypoints])
    self.model.waypoints.clear()    # clear the actual waypoint positions after deleting them on the view!

    self.collecting_waypoints = True
    self.model.sort_agents_by_angle()
    self.agent_no = 0
    self.model.highlight_agent(0)


  def collect_waypoints(self, event):
    if event.type != pygame.MOUSEBUTTONDOWN:
      return

    position = self.view.pixel_belongs_to_block(event.pos)
    self.model.select_square(position)
    
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

    #if event.key == pygame.K_c:
    #  self.model.waypoints.clear()

    #if event.key == pygame.K_w:
    #  self.start_collecting_waypoints()

