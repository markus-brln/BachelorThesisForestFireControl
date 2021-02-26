
import pygame

COLOR = { "BLACK": (0, 0, 0), 
          "WHITE": (255, 255, 255),
          "RED": (255, 0,0 )}
## TODO:
#   Fire Propagation

class Model:
  # Other parameters can be added later
  ## Length: Grid size
  ## Agents: Number of agents
  def __init__(self, length, agents):
    self.size = length
    self.agents = agents
    self.selected_squares = [ (5, 5)]

    self.startEpisode()
    self.terminal_state = False
  

  def startEpisode(self):
    self.selected_squares = [ (5, 5)]
    self.time = 0
    self.firepos = [(self.size / 2, self.size / 2)]


  def time_step(self):
    self.time += 1

  def select_square(self, width, height):
    self.selected_squares += [(width, height)]
  


## Separate these classes, figure out how to import code from other file
class View:
  def __init__(self, model, grid_block_size):
    self.model = model
    self.grid_block_size = grid_block_size           ## Width and height of block in grid
    self.window_size = model.size * grid_block_size

    self.window = pygame.display.set_mode((self.window_size, self.window_size))

    self.clock = pygame.time.Clock() ## Don't know if necessary yet
  

  def update(self):
    self.window.fill(COLOR["BLACK"])
    self.draw_grid()
    pygame.display.update()


  def draw_grid(self):
    block_size = self.grid_block_size
    for x in range(self.model.size):
      for y in range(self.model.size):
        rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
        pygame.draw.rect(self.window, COLOR["WHITE"], rect, 1)
    

  def show_selected(self,block_size):
    block_size = self.grid_block_size
    for selected in self.model.selected_squares:
        rect = pygame.Rect(selected[0] * block_size - block_size / 2, selected[1] * block_size, block_size, block_size)
        pygame.draw.rect(self.window, COLOR["RED"], rect, block_size)
    

def main():
  environment = Model(20, [])
  view = View(environment, 15)
  while(True):
    view.update()


if __name__=="__main__":
  main()