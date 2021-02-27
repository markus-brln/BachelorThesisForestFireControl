### Required packages
# Pygame: pip install pygame

import pygame

pygame.init()  ## Initialize Pygame 

class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, agents):
    self.size = length
    self.agents = agents

    self.startEpisode()           # Initialize episode

  
  def startEpisode(self):
    self.selected_squares = set() # Reset selection
    self.time = 0                 # Reset time
    self.terminal_state = False   # Restart so at initial state

    # Start fire in the middle of the map
    self.firepos = set([(int(self.size / 2), int(self.size / 2))])


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
    
    
  ## TODO
  def expand_fire(self):
    pass


  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    print("Shutting Down")


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
    self.mouse_button_pressed = False
    self.last_clicked = (-1, 0)
    
  
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
    if event.key == pygame.K_RETURN:
      self.model.startEpisode()       ## Return / ENTER to go to next episode

    # Update the view
    self.view.update()


def main():
  environment = Model(25, [])         ## Initialize Environment
  view = View(environment, 25)        ## Start View
  controller = Controller(environment, view) ## Initialize Controller with model and view
  while(True):
    controller.update(pygame.event.get())    ## Let the controller take over


if __name__=="__main__":
  main()