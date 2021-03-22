import pygame
import time
from Model.environment import Model
from View.view import View, UpdateType

class Controller:
  def __init__(self, model: Model, view: View):
    self.model = model
    self.view = view

    # Initialization
    self.mouse_button_pressed = False   ## Mouse button assumed not to be pressed initially

  def update(self, event):

    if self.model.reset_necessary:        # M update view when resetting env (hacky way)
      self.view.update()
      self.model.reset_necessary = False

    #for event in pygame_events:
    if event.type == pygame.QUIT:
      # Exit the program
      self.shut_down(event)
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
    if self.mouse_button_pressed == 1: ## Left click
      self.model.select_square(position)
    else:                              ## Right click
      self.model.deselect_square(position)

  def key_press(self, event):
    start = time.time()
    if event.key == pygame.K_SPACE:
      self.model.time_step()          ## Space to go to next timestep
      return

    if event.key == pygame.K_RETURN:
      self.model.start_episode()       ## Return / ENTER to go to next episode

    if event.key == pygame.K_c:
      self.model.waypoints.clear()