import pygame

## Needs access to model
from Model.environment import Model
from enum import Enum


class UpdateType(Enum):
  ALL = 0
  AGENTS = 1
  FIRE = 2
  FIREBREAKS = 3
  WAYPOINT = 4


pygame.font.init() # add text
myfont = pygame.font.SysFont('arial', 22)



## Separate these classes, figure out how to import code from other file
class View:
  # Model: The model to display
  # grid_block_size: The size of a block in the gridworld in pixels
  def __init__(self, model: Model, grid_block_size: int):
    self.model = model
    self.grid_block_size = grid_block_size           ## Width and height of block in grid
    window_size = model.size * grid_block_size
    self.window = pygame.display.set_mode((window_size, window_size))
    programIcon = pygame.image.load('View/fire.png')
    pygame.display.set_icon(programIcon)
    self.clock = pygame.time.Clock() ## Don't know if necessary yet, or what it actually does

    # Does nothing yet, for zooming functionality
    self.translation = (0, 0)
    self.scale = 1



  def update(self, updateType = None):
    #print("updating")

    if not updateType:        # no type specified -> update everything
      # Draw Background and grid
      self.window.fill(pygame.Color("ForestGreen"))
      self.draw_grid()
      self.draw_fire()        # draw both new and old fire
      self.draw_edge_fire()
      self.draw_firebreaks()
      self.draw_waypoints()
      self.draw_agents()
      self.draw_mouse_coords()
      # Update pygame display
      pygame.display.update()
      return

    if UpdateType.FIRE in updateType:
      self.draw_edge_fire()

    if UpdateType.WAYPOINT in updateType:
      self.draw_waypoints()

    if UpdateType.AGENTS in updateType:
      self.draw_agents()

    pygame.display.update()



  # Draw the lines separating the blocks in the grid
  def draw_grid(self):
    block_size = self.grid_block_size
    for x in range(self.model.size):
      for y in range(self.model.size):
        rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
        pygame.draw.rect(self.window, pygame.Color("White"), rect, 1)


  # Fill the blocks at the provided positions with the provided color
  def fill_blocks(self, positions, color):
    #print("filling blocks")
    block_size = self.grid_block_size
    for pixels in positions:
      ## I really do not want to do it like this TODO possibly change
      for x in range(pixels[0] * block_size, (pixels[0] + 1) * block_size):
        for y in range(pixels[1] * block_size, (pixels[1] + 1) * block_size):
          rect = (x, y, 1, 1)
          pygame.draw.rect(self.window, color, rect, 1)

  # Draw the agent positions blue
  def draw_agents(self):
    self.fill_blocks(self.model.agent_positions(), pygame.Color("DarkBlue"))


  def draw_fire(self):
    """Fill all fire blocks."""
    self.fill_blocks(self.model.firepos, pygame.Color("Red"))


  def draw_edge_fire(self):
    """Only fill the newly ignited blocks."""
    self.fill_blocks(self.model.firepos_edge, pygame.Color("Red"))


  def draw_firebreaks(self):
    self.fill_blocks(self.model.firebreaks, pygame.Color("SaddleBrown"))

  # Draw the waypoints positions black
  def draw_waypoints(self):
    self.fill_blocks(self.model.waypoints, pygame.Color("Black"))
  
  ## Determine in which block a pixel lies
  def pixel_belongs_to_block(self, pos):
    x = int(pos[0] / self.grid_block_size)
    y = int(pos[1] / self.grid_block_size)
    return (x, y)

  def draw_mouse_coords(self):
    ## keep track of mouse position
    position = pygame.mouse.get_pos()
    pos = self.pixel_belongs_to_block(position)
    textsurface = myfont.render(str(pos), False, (0, 0, 200))
    self.window.blit(textsurface, (0, 0))