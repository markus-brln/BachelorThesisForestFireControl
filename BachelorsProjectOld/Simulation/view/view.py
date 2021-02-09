import pygame

from .constants import WIDTH, HEIGHT


def draw(board, world_new):
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("FOREST FIRE SIMULATION")
    board.draw_state(WIN, world_new)
    pygame.display.update()


def stop():
    pygame.display.quit()
    pygame.quit()
