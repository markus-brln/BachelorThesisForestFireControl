import pygame

from View.updatetype import UpdateType

## Needs access to model
from Model.model import Model
from Model.node import NodeState
from enum import Enum

pygame.font.init() # add text
myfont = pygame.font.SysFont('arial', 22)


## Separate these classes, figure out how to import code from other file
class View:
  # Model: The model to display
  # grid_block_size: The size of a block in the gridworld in pixels
  def __init__(self, model: Model, grid_block_size: int):
    ## Store Parameters
    self.model = model
    self.grid_block_size = grid_block_size           ## Width and height of block in grid
    window_size = model.size * grid_block_size
    self.window = pygame.display.set_mode((window_size, window_size))
    programIcon = pygame.image.load('View/fire.png')
    pygame.display.set_icon(programIcon)
    self.clock = pygame.time.Clock() ## Don't know if necessary yet, or what it actually does
    
    self.draw_initial_model()     ## Draw initial model
    self.model.subscribe(self)    ## Subscribe to changes

    self.highlighted_agent = None


  def draw_agents(self):
    for agent in self.model.agents:
      if not agent.dead:
        self.draw_block(agent.position, pygame.Color("DarkBlue"))
      
    if self.highlighted_agent is not None:
      self.draw_block(self.highlighted_agent.position, pygame.Color("Yellow"))
    


  def draw_initial_model(self):
      self.window.fill(pygame.Color("ForestGreen"))
      # self.draw_grid()
      for agent in self.model.agents:
        self.draw_block(agent.position, pygame.Color("DarkBlue"))
      self.draw()


  ## Determine in which block a pixel lies
  def pixel_belongs_to_block(self, pos):
    x = int(pos[0] / self.grid_block_size)
    y = int(pos[1] / self.grid_block_size)
    return x, y

  
  def update(self, update_type: UpdateType, position = None, node = None, agent = None):
    if update_type == UpdateType.RESET:
      self.draw_initial_model()

    if update_type == UpdateType.NODE:
      self.node_change(node)
      
    if update_type == UpdateType.AGENT:
      self.node_change(agent.prev_node)
      self.draw_block(agent.position, pygame.Color("DarkBlue"))

    if update_type == UpdateType.WAYPOINT:
      self.draw_block(position, pygame.Color("Black"))
      self.draw()
    
    if update_type == UpdateType.HIGHLIGHT_AGENT:
      if self.highlighted_agent is not None:
        self.draw_block(self.highlighted_agent.position,  pygame.Color("DarkBlue"))
      if agent is not None:
        self.draw_block(agent.position, pygame.Color("Yellow"))
      self.highlighted_agent = agent
      self.draw()

    if update_type == UpdateType.TIMESTEP_COMPLETE:
      self.draw()


  def node_change(self, node):
    if node.state == NodeState.NORMAL:
      colour = pygame.Color("ForestGreen")
    if node.state == NodeState.FIREBREAK:
      colour = pygame.Color("SaddleBrown")
    if node.state == NodeState.ON_FIRE:
      colour = pygame.Color("Red")
    if node.state == NodeState.BURNED_OUT:
      colour = pygame.Color("DarkGrey")

    self.draw_block(node.position, colour)


  def draw_block(self, position, colour):
    block_size = self.grid_block_size
    for x in range(position[0] * block_size, (position[0] + 1) * block_size):
      for y in range(position[1] * block_size, (position[1] + 1) * block_size):
        rect = (x, y, 1, 1)
        pygame.draw.rect(self.window, colour, rect, 1)


  # Draw the lines separating the blocks in the grid
  def draw_grid(self):
    block_size = self.grid_block_size
    for x in range(self.model.size):
      for y in range(self.model.size):
        rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
        pygame.draw.rect(self.window, pygame.Color("White"), rect, 1)


  def draw_mouse_coords(self):
    ## keep track of mouse position
    position = pygame.mouse.get_pos()
    pos = self.pixel_belongs_to_block(position)
    textsurface = myfont.render(str(pos), False, (0, 0, 200))
    self.window.blit(textsurface, (0, 0))

  
  def clear_waypoints(self, old_waypoints):
    for node in old_waypoints:
      self.node_change(node)
    self.draw_agents()

  
  def draw(self):
    self.draw_block(self.model.centre, (255,255,255))
    pygame.display.update()