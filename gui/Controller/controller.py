import pygame
import time
from Model.model import Model
from View.view import View
from Model.utils import *

## Enum holding the different ways the model can be controlled
class Mode(Enum):
  DATA_GENERATION = 0,
  CNN = 1,
  OLD_CNN = 2,
  ETCETERA = 3,


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
    print("Assigning waypoints")
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


class NN_Controller:
  def __init__(self, filename, model: Model):
    self.load_NN(filename)
    self.model = model
  

  def run(self, iterations, timesteps = 20):
    for _ in range(iterations):
      self.model.start_episode()
      while self.model.firepos != set(): # While firepos not empty
        NN_output = self.predict()       # Get NN output
        self.steer_model(NN_output)      # use output to assign waypoints
        for _ in range(timesteps):
          self.model.timestep()
        time.sleep(1) # So we can see what's going on. Disable when running.


  def steer_model(self, nn_output):
    pass ## Assign waypoints here

  def load_NN(self, filename):
    pass ## will be quicker with a bit of help


  def predict(self):
    pass

