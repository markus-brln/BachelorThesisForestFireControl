import pygame
from Simulation.view.constants import *

from Simulation.utility import (
    layer,
    types,
)

'''
This class is used to draw the board. It uses pygame to to so. 
'''


class Board:
    def __init__(self, world):
        self.board = world
        self.win = self.start()
        self.font_1 = None
        self.font_2 = None
        self.font_3 = None
        self.surface_1 = None
        self.surface_2 = None
        self.surface_3 = None
        self.surface_4 = None
    '''
    Draw the current state of the environment
    '''

    def draw_state(self, win, world_new):
        self.update(win, world_new)
        for x in range(ROWS):
            for y in range(COLS):
                if self.board.env[x, y, layer['type']] == types['grass']:
                    pygame.draw.rect(win, GREEN, (y * SQUARE_SIZE, x * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                pygame.draw.rect(win, (142,210,134), (y * SQUARE_SIZE, x * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)

                if self.board.env[x, y, layer['type']] == types['fire'] == 1:
                    pygame.draw.rect(win, RED, (y * SQUARE_SIZE, x * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                if self.board.env[x, y, layer['type']] == types['dirt']:
                    pygame.draw.rect(win, BROWN, (y * SQUARE_SIZE, x * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                if self.board.env[x, y, layer['agent_pos']] == 1 or self.board.env[x, y, layer['other_pos']] == 1:
                    pygame.draw.rect(win, ANTRACIET, (y * SQUARE_SIZE, x * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                    win.blit(self.surface_1, (y * SQUARE_SIZE + 8, x * SQUARE_SIZE + 7))


        self.draw_text(win)

    '''
    Sets the window in which the environment will be drawn 
    '''

    def start(self):
        WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("FOREST FIRE SIMULATION")
        return WIN

    '''
    Draws state and updates the display
    '''

    def draw(self, board, world_new):
        board.draw_state(self.win, world_new)
        pygame.display.update()

    '''
    closes the display
    '''

    def stop(self):
        pygame.display.quit()
        pygame.quit()

    '''
    Updates the board (environment), fills the background and initializes the font
    '''

    def update(self, win, world_new):
        self.board = world_new
        win.fill(BLACK)
        pygame.font.init()
        self.font_1 = pygame.font.SysFont('Comic Sans', 40)
        self.font_2 = pygame.font.SysFont('Comic Sans', 40)
        self.font_3 = pygame.font.SysFont('Comic Sans', int(90 / (self.board.WIDTH / 11)))
        self.surface_1 = self.font_3.render("A", False, WHITE)
        self.surface_2 = self.font_2.render("EPISODE: " + str(self.board.performance.episode), False, OFFWHITE)
        self.surface_3 = self.font_1.render("FIRES ISOLATED: " + str(self.board.performance.amount_fires_isolated) + "/"
                                            + str(self.board.performance.episode), False, OFFWHITE)
        self.surface_4 = self.font_1.render("AVERAGE PERCENTAGE OF BURNT MAP: " +
                                            str(round(self.board.performance.cumulative_burnt /
                                                      (self.board.performance.episode + 1), 2)), False, OFFWHITE)

    '''
    Draws the text 
    '''

    def draw_text(self, win):
        pygame.draw.rect(win, BLACK, (0, 0, 400, 100))
        win.blit(self.surface_3, (20, 45))
        # win.blit(self.surface_4, (20, 60))
        win.blit(self.surface_2, (20, 15))



