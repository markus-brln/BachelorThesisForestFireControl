### Required packages
# Pygame: pip install pygame

import pygame

pygame.init()  ## Initialize Pygame 

class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, agents, firesize = 1):
    self.size = length
    self.firesize = firesize
    self.agents = agents

    self.startEpisode()           # Initialize episode

  
  def startEpisode(self):
    self.selected_squares = set() # Reset selection
    self.time = 0                 # Reset time
    self.terminal_state = False   # Restart so at initial state

    self.centreX = int(self.size / 2)
    self.centreY = int(self.size / 2)
    # Start fire in the middle of the map
    self.firepos = set()
    self.firepos.clear()
    self.firepos.add((self.centreX, self.centreY))


    idx = 1
    while idx < self.firesize:
     self.fire = list(self.firepos)
     for pos in self.fire:
      neighbours = self.getNeighbours(pos)
      for neighbour in neighbours:
       self.firepos.add(neighbour)
     idx += 1


  ## Increment the time by one.
  # TODO: possibly reset selection?
  def time_step(self):
    self.time += 1                # Increment time
    self.expand_fire()            # Determine fire propagation


  def select_square(self, position):
    if position not in self.firepos:       ## Cannot set waypoint in the fire
      self.selected_squares.add(position)  # Add to list of waypoints
    

  def deselect_square(self, position):
    self.selected_squares.discard(position) # Remove from list of waypoints
    
  def getNeighbours(self, position):
    self.pos = position
    x = self.pos[0]
    y = self.pos[1]
    self.left = x - 1
    self.right = x + 1
    self.top = y + 1
    self.bottom = y - 1
    ## new list of neighbouring cells
    neighbours = [(self.left, y), (x, self.top), (self.right, y), (x, self.bottom)]
    for neighbour, (a, b) in enumerate(neighbours):
     if self.position_in_bounds((a, b)):
      return neighbours
     else:
      self.shut_down()


  ## currently stops when the fire reaches the edge of the map for simplicity but
  ## also as it makes it impossible for the agent to contain the fire
  def expand_fire(self):
    self.fire = list(self.firepos)
    print(self.fire)
    for pos in self.fire:
      neighbours = self.getNeighbours(pos)
      for neighbour in neighbours:
        self.firepos.add(neighbour)

  def position_in_bounds(self, position):
    self.pos = position
    self.x = self.pos[0]
    self.y = self.pos[1]
    # print("x: ", self.x, "y: ", self.y)
    if ((self.x >= 0 & self.x <= self.size) & (self.y >= 0 & self.y <= self.size)):
      return True
    else:
      return False


  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    print("Fire out of control!")
    self.startEpisode()


## Separate these classes, figure out how to import code from other file
class View:
  # Model: The model to display
  # grid_block_size: The size of a block in the gridworld in pixels
  def __init__(self, model: Model, grid_block_size: int):
    self.model = model
    self.grid_block_size = grid_block_size           ## Width and height of block in grid
    window_size = model.size * grid_block_size
    self.window = pygame.display.set_mode((window_size, window_size))

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
    self.draw_selected()
    self.draw_fire()
    self.draw_agents()

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
    for selected in positions:
      ## I really do not want to do it like this TODO possibly change
      for x in range(selected[0] * block_size, (selected[0] + 1) * block_size):
        for y in range(selected[1] * block_size, (selected[1] + 1) * block_size):
          rect = (x, y, 1, 1)
          pygame.draw.rect(self.window, color, rect, 1)

  # Draw the agent positions blue
  def draw_agents(self):
    self.fill_blocks(self.model.agents, pygame.Color("Cyan"))

  # Draw the fire positions red
  def draw_fire(self):
    self.fill_blocks(self.model.firepos, pygame.Color("FireBrick"))

  # Draw the selected positions black
  def draw_selected(self):
    self.fill_blocks(self.model.selected_squares, pygame.Color("Black"))
  
  ## Determine in which block a pixel lies
  # TODO: change after zooming
  def pixel_belongs_to_block(self, pos):
    x = int(pos[0] / self.grid_block_size)
    y = int(pos[1] / self.grid_block_size)
    return (x, y)


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
      elif event.type == pygame.KEYDOWN:
        # Keyboard button pressed
        self.key_press(event)
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
      self.model.startEpisode()       ## Return / ENTER to go to next episode
    # Update the view
    self.view.update()


def main():
  starting_firesize = 1
  environment = Model(9, starting_firesize, [])         ## Initialize Environment
  view = View(environment, 20)        ## Start View
  controller = Controller(environment, view) ## Initialize Controller with model and view
  while(True):
    controller.update(pygame.event.get())    ## Let the controller take over


if __name__=="__main__":
  main()