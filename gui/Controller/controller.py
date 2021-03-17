import pygame

from Model.environment import Model
from View.view import View

class Controller:
  def __init__(self, model: Model, view: View):
    self.model = model
    self.view = view

    # Initialization
    self.mouse_button_pressed = False   ## Mouse button assumed not to be pressed initially

  def update(self, pygame_events):
    for event in pygame_events:
      if event.type == pygame.QUIT:
        # Exit the program
        self.shut_down(event)
      elif event.type == pygame.MOUSEBUTTONDOWN:
        # Mouse stationary and mouse button pressed
        self.mouse_button_pressed = event.button
        self.select(event.pos)
      elif event.type == pygame.MOUSEBUTTONUP:
        # Mouse button released
        self.mouse_button_pressed = False
        self.last_clicked = (-1, 0)           ## Reset to allow clicking a square twice in a row
      elif event.type == pygame.MOUSEMOTION and self.mouse_button_pressed:
        # Mouse button pressed and dragged
        self.select(event.pos)
      elif event.type == pygame.KEYDOWN:
        # Keyboard button pressed
        self.key_press(event)
      
  def shut_down(self, event):
    self.model.shut_down()              ## Ensure proper shutdown of the model
    exit(0)                             ## Exit program
  
  def select(self, position):
    # Determine the block the mouse is covering
    position = self.view.pixel_belongs_to_block(position)
    
    # Select or deselect that block 
    if self.mouse_button_pressed == 1: ## Left click
      self.model.select_square(position)
    else:                              ## Right click
      self.model.deselect_square(position)

    # Update the view
    self.view.update()


  def key_press(self, event):
    if event.key == pygame.K_SPACE:
      self.model.time_step()          ## Space to go to next timestep
    ##TODO possibly add a revert time step option to go back one
    if event.key == pygame.K_RETURN:
      self.model.start_episode()       ## Return / ENTER to go to next episode
    if event.key == pygame.K_c:
      self.model.waypoints.clear()

    # Update the view
    self.view.update()