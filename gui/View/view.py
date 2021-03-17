import pygame

## Needs access to model
from Model.environment import Model


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

    self.update()
  

  def update(self):
    # Draw Background and grid
    self.window.fill(pygame.Color("ForestGreen"))
    self.draw_grid()
    # State Dependent drawing
    self.draw_fire()
    self.draw_firebreaks()
    self.draw_waypoints()
    self.draw_agents()
    self.draw_mouse_coords()
    # Update pygame display
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
    block_size = self.grid_block_size
    for waypoints in positions:
      ## I really do not want to do it like this TODO possibly change
      for x in range(waypoints[0] * block_size, (waypoints[0] + 1) * block_size):
        for y in range(waypoints[1] * block_size, (waypoints[1] + 1) * block_size):
          rect = (x, y, 1, 1)
          pygame.draw.rect(self.window, color, rect, 1)

  # Draw the agent positions blue
  def draw_agents(self):
    self.fill_blocks(self.model.agent_positions(), pygame.Color("DarkBlue"))

  # Draw the fire positions red
  def draw_fire(self):
    fire = list(self.model.firepos)
    self.fill_blocks(fire, pygame.Color("Red"))


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