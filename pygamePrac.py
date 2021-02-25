import pygame

# Intialize the pygame
pygame.init()

# create the screen
Background = pygame.display.set_mode((1000, 800))
Background.fill((0, 220, 0))

for i in range(0, 1000, 4):
    pygame.draw.line(Background, (255, 255, 255), (0, i), (1000, i))
    pygame.draw.line(Background, (255, 255, 255), (i, 0), (i, 800))

# Game Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for i in range(0, 100000, 1):
        fire = pygame.Rect(500, 400, 2+i/10, 2+i/10)
        pygame.draw.rect(Background, (200, 0, 0), fire)
        pygame.display.update()

    pygame.display.update()