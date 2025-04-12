import pygame
import random
import time

# Funkcja do losowego punktu w zakresie
def random_point_in_range(x_min, x_max, y_min, y_max):
    return random.randint(x_min, x_max), random.randint(y_min, y_max)

# Ustawienia ekranu (mapy)
square_size = 500  # Rozmiar okna
circle_center = random_point_in_range(100, 400, 100, 400)  # Losowy środek strefy
circle_radius = random.randint(50, 150)  # Losowy promień strefy

# Inicjalizacja pygame
pygame.init()
screen = pygame.display.set_mode((square_size, square_size))
pygame.display.set_caption("Symulacja strefy i punktów")
clock = pygame.time.Clock()

# Kolory
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Główna pętla symulacji
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Wyczyszczenie ekranu
    screen.fill(WHITE)

    # Rysowanie strefy
    pygame.draw.circle(screen, BLUE, circle_center, circle_radius)

    # Losowy punkt
    point = random_point_in_range(0, square_size, 0, square_size)

    # Sprawdzenie, czy punkt jest w strefie
    distance = ((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)**0.5
    if distance <= circle_radius:
        message = "Punkt wewnątrz strefy."
        point_color = GREEN  # Zielony, jeśli w strefie
    else:
        message = "Dziadek ucieka!!!!!"
        point_color = RED  # Czerwony, jeśli poza strefą

    print(message)

    # Rysowanie punktu
    pygame.draw.circle(screen, point_color, point, 10)  # Rozmiar punktu = 10

    # Aktualizacja ekranu
    pygame.display.flip()

    # Opóźnienie
    time.sleep(3)
    clock.tick(60)

# Zakończenie pygame
pygame.quit()
