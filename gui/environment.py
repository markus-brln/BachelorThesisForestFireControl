import pygame

from time import sleep ## To redraw once a second. Should not be necessary
                       ## after implementing redrawing after updates.


pygame.init()  ## Initialize Pygame 

COLOR = { "BLACK": (0, 0, 0), 
          "WHITE": (255, 255, 255),
          "RED": (255, 0,0 ),
          "GREEN": (0, 180, 0)
          }

class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: TODO determine: Number of agents or tuples of agent positions
  def __init__(self, length: int, agents):
    self.size = length
    self.agents = agents
    self.selected_squares = [ (5, 5)]

    self.startEpisode()
    self.terminal_state = False
    self.subscribers = []

  
  def startEpisode(self):
    self.selected_squares = [ (5, 5)]
    self.time = 0
    self.firepos = [(self.size / 2, self.size / 2)]


  def time_step(self):
    self.time += 1
    self.expand_fire()


  def select_square(self, width, height):
    if (width, height) in self.selected_squares:
      self.selected_squares.remove((width, height))
    else:
      self.selected_squares += [(width, height)]


  ## TODO
  def expand_fire(self):
    pass


  ## TODO: e.g. save data and ensure proper exiting of program
  def shut_down(self):
    print("Shutting Down")


## Separate these classes, figure out how to import code from other file
class View:
  def __init__(self, model: Model, grid_block_size: int):
    model.view = self
    self.model = model
    self.grid_block_size = grid_block_size           ## Width and height of block in grid
    self.window_size = model.size * grid_block_size

    self.window = pygame.display.set_mode((self.window_size, self.window_size))

    self.clock = pygame.time.Clock() ## Don't know if necessary yet, or what it actually does
    self.update()
  

  def update(self):
    self.window.fill(COLOR["GREEN"])
    self.draw_grid()
    self.draw_selected()
    self.draw_fire()
    self.draw_agents()
    pygame.display.update()


  def draw_grid(self):
    block_size = self.grid_block_size
    for x in range(self.model.size):
      for y in range(self.model.size):
        rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
        pygame.draw.rect(self.window, COLOR["WHITE"], rect, 1)


  def draw_agents(self):
    pass


  def draw_fire(self):
    pass
    

  def draw_selected(self):
    block_size = self.grid_block_size
    for selected in self.model.selected_squares:
      ## I really do not want to do it like this
      for x in range(selected[0] * block_size, (selected[0] + 1) * block_size):
        for y in range(selected[1] * block_size, (selected[1] + 1) * block_size):
          rect = (x, y, 1, 1)
          pygame.draw.rect(self.window, COLOR["RED"], rect, 1)


class Controller:
  def __init__(self, model: Model, view: View):
    self.model = model
    self.view = view
    
  
  def update(self, pygame_events):
    for event in pygame_events:
      if event.type == pygame.QUIT:
        self.shut_down(event)
      elif event.type == pygame.MOUSEBUTTONDOWN:
        self.click(event)
      
  def shut_down(self, event):
    self.model.shut_down()
    exit(0)

  
  def click(self, event):
    (x, y) = event.pos
    x = int(x / self.view.grid_block_size)
    y = int(y / self.view.grid_block_size)
    self.model.select_square(x, y)
    self.view.update()


def main():
  environment = Model(20, [])
  view = View(environment, 25)
  controller = Controller(environment, view)
  while(True):
    controller.update(pygame.event.get())


if __name__=="__main__":
  main()