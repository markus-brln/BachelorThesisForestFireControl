import pygame

from time import sleep ## To redraw once a second. Should not be necessary
                       ## after implementing redrawing after updates.


pygame.init()  ## Initialize Pygame 

class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, agents):
    self.size = length
    self.agents = agents
    self.selected_squares = []

    self.startEpisode()
    self.terminal_state = False
    self.subscribers = []

  
  def startEpisode(self):
    self.selected_squares = []
    self.time = 0
    self.firepos = [(int(self.size / 2), int(self.size / 2))]


  def time_step(self):
    self.time += 1
    self.expand_fire()


  def select_square(self, position):
    if position in self.selected_squares:
      self.selected_squares.remove(position)
    else:
      self.selected_squares += [position]
    
  ## TODO
  def expand_fire(self):
    pass


  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    print("Shutting Down")


## Separate these classes, figure out how to import code from other file
class View:
  def __init__(self, model: Model, grid_block_size: int):
    self.model = model
    self.grid_block_size = grid_block_size           ## Width and height of block in grid
    self.window_size = model.size * grid_block_size
    self.window = pygame.display.set_mode((self.window_size, self.window_size))

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


  def draw_grid(self):
    block_size = self.grid_block_size
    for x in range(self.model.size):
      for y in range(self.model.size):
        rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
        pygame.draw.rect(self.window, pygame.Color("black"), rect, 1)


  def fill_blocks(self, positions, color):
    block_size = self.grid_block_size
    for selected in positions:
      ## I really do not want to do it like this TODO possibly change
      for x in range(selected[0] * block_size, (selected[0] + 1) * block_size):
        for y in range(selected[1] * block_size, (selected[1] + 1) * block_size):
          rect = (x, y, 1, 1)
          pygame.draw.rect(self.window, color, rect, 1)


  def draw_agents(self):
    self.fill_blocks(self.model.agents, pygame.Color("Cyan"))


  def draw_fire(self):
    self.fill_blocks(self.model.firepos, pygame.Color("FireBrick"))

  def draw_selected(self):
    self.fill_blocks(self.model.selected_squares, pygame.Color("Black"))
  

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
        self.shut_down(event)
      elif event.type == pygame.MOUSEMOTION and self.mouse_button_pressed:
        self.select(event.pos)
      elif event.type == pygame.KEYDOWN:
        self.key_press(event)
      elif event.type == pygame.MOUSEBUTTONDOWN:
        self.mouse_button_pressed = True
        self.select(event.pos)
      elif event.type == pygame.MOUSEBUTTONUP:
        self.mouse_button_pressed = False
      
  def shut_down(self, event):
    self.model.shut_down()
    exit(0)
  
  def select(self, position):
    position = self.view.pixel_belongs_to_block(position)
    if position == self.last_clicked:
      return
    
    self.model.select_square(position)
    self.last_clicked = position
    self.view.update()


  def key_press(self, event):
    if event.key == pygame.K_SPACE:
      self.model.time_step()
    if event.key == pygame.K_RETURN:
      self.model.startEpisode()

    self.view.update()


def main():
  environment = Model(25, [])
  view = View(environment, 25)
  controller = Controller(environment, view)
  while(True):
    controller.update(pygame.event.get())


if __name__=="__main__":
  main()