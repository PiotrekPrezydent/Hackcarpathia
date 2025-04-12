import pygame
import random
import math
import time

pygame.init()
screen_size = 600
screen = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption("Dziadek vs Policjant")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

WHITE = (255, 255, 255)
GRAY = (230, 230, 230)
GREEN = (0, 255, 0, 80)
YELLOW = (255, 255, 0, 80)
BLUE = (0, 100, 255, 60)  # Niebieska strefa
RED = (255, 0, 0)  # Dziadek
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

# Pozycje
center = (screen_size // 2, screen_size // 2)
grandpa_pos = [center[0], center[1]]
grandpa_speed = 3.5  # Dziadek szybszy niż policjant
policeman_pos = [random.randint(0, screen_size), random.randint(0, screen_size)]
policeman_speed = 2.5  # Szybkość policjanta
last_direction_change = time.time()
last_target_change = time.time()
time_in_house = None  # Czas, kiedy dziadek jest w domu na 3 sekundy
policeman_respawn_time = None  # Czas, kiedy policjant został zrespiony
last_teleport_time = 0  # Czas ostatniej teleportacji


# Losowanie nowego celu dla dziadka
def random_target():
    return [random.randint(0, screen_size), random.randint(0, screen_size)]


def random_police_position(away_from_pos, min_dist=200):
    while True:
        pos = [random.randint(0, screen_size), random.randint(0, screen_size)]
        if math.hypot(pos[0] - away_from_pos[0], pos[1] - away_from_pos[1]) >= min_dist:
            return pos


def is_in_zone(position, radius):
    """Sprawdza, czy pozycja znajduje się w strefie"""
    return math.hypot(position[0] - center[0], position[1] - center[1]) <= radius


def move_away_from_zone(position, radius, speed):
    """Porusza policjanta z dala od strefy, w kierunku najbliższej krawędzi"""
    if is_in_zone(position, radius):
        # Liczymy wektor przeciwny do kierunku w stronę strefy, aby oddalić się od niej
        angle = math.atan2(position[1] - center[1], position[0] - center[0])
        # Poruszamy się w kierunku najbliższej krawędzi
        if position[0] < center[0]:
            return [speed, 0]  # Idziemy w prawo
        elif position[0] > center[0]:
            return [-speed, 0]  # Idziemy w lewo
        elif position[1] < center[1]:
            return [0, speed]  # Idziemy w dół
        elif position[1] > center[1]:
            return [0, -speed]  # Idziemy w górę
    return [0, 0]  # Brak potrzeby zmiany kierunku


# Rysowanie
def draw_stars(screen, count):
    for i in range(count):
        x = screen_size - 30 * (count - i)
        y = 10
        shadow = font.render("*", True, BLACK)
        screen.blit(shadow, (x + 1, y + 1))
        star = font.render("*", True, ORANGE)
        screen.blit(star, (x, y))


def draw_filled_circle(color, position, radius):
    surface = pygame.Surface((screen_size, screen_size), pygame.SRCALPHA)
    pygame.draw.circle(surface, color, position, radius)
    screen.blit(surface, (0, 0))


def change_direction():
    angle = random.uniform(0, 2 * math.pi)
    return [math.cos(angle), math.sin(angle)]


def move_toward(src, dst, speed):
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return src
    return [src[0] + dx / dist * speed, src[1] + dy / dist * speed]


def teleport_if_necessary(pos):
    global last_teleport_time
    # Dziadek przechodzi przez krawędź tylko, jeśli jest za blisko krawędzi i nie jest zablokowana teleportacja
    edge_distance = 10  # Odległość od krawędzi, w której może dojść do teleportacji
    current_time = time.time()

    if current_time - last_teleport_time >= 2:  # Jeśli minęły 2 sekundy od ostatniej teleportacji
        if pos[0] < edge_distance:
            pos[0] = screen_size  # Teleportacja na przeciwną stronę ekranu
        elif pos[0] > screen_size - edge_distance:
            pos[0] = 0
        if pos[1] < edge_distance:
            pos[1] = screen_size
        elif pos[1] > screen_size - edge_distance:
            pos[1] = 0

        last_teleport_time = current_time  # Aktualizuj czas ostatniej teleportacji

    return pos


# Inicjalizacja zmiennej grandpa_target
grandpa_target = random_target()

# Zmienna ścigania policjanta
policeman_chasing = False

running = True
while running:
    screen.fill(GRAY)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Rysuj strefy
    draw_filled_circle(GREEN, center, 50)
    draw_filled_circle(YELLOW, center, 100)
    draw_filled_circle(BLUE, center, 150)  # Niebieska strefa

    # Jeśli dziadek jest złapany przez policjanta, teleportuj go do domu na 3 sekundy
    distance_to_policeman = math.hypot(grandpa_pos[0] - policeman_pos[0], grandpa_pos[1] - policeman_pos[1])
    if distance_to_policeman < 15:
        # Zatrzymanie policjanta, zniknięcie na chwilę, respawn w losowym miejscu
        policeman_respawn_time = time.time()
        policeman_pos = random_police_position(grandpa_pos, min_dist=200)  # Nowa pozycja dla policjanta
        policeman_speed = 1.5  # Policjant porusza się wolniej po respawnie
        time_in_house = time.time()  # Zapisujemy czas, kiedy dziadek trafił do domu

    if time_in_house is not None:
        # Dziadek jest teraz w domu przez 3 sekundy
        if time.time() - time_in_house < 3:
            grandpa_pos = center  # Dziadek jest w "domu" (środek ekranu)
        else:
            # Po 3 sekundach dziadek wraca do gry
            time_in_house = None
            grandpa_target = random_target()  # Resetuj punkt celu
            grandpa_pos = move_toward(grandpa_pos, grandpa_target, grandpa_speed)  # Wróć do normalnego ruchu
    else:
        # Sprawdzamy, czy dziadek opuścił strefę
        in_zone = False
        for radius in [50, 100, 150]:
            if is_in_zone(grandpa_pos, radius):
                in_zone = True
                break

        # Policjant zaczyna gonić, gdy dziadek wyjdzie ze strefy
        if not in_zone:
            if not policeman_chasing:  # Zaczyna gonić, jeśli jeszcze nie goni
                policeman_chasing = True
                print("Policjant zaczyna gonić dziadka!")

            grandpa_speed = 5  # Zwiększenie prędkości dziadka, kiedy policjant jest blisko
        else:
            policeman_chasing = False
            grandpa_speed = 3.5  # Normalna prędkość dziadka

        # Losowy ruch dziadka – tylko kilka kroków do celu
        if time.time() - last_direction_change > 2:  # Zmieniamy punkt docelowy co 2 sekundy
            grandpa_target = random_target()  # Losowanie nowego celu
            last_direction_change = time.time()

        # Sprawdzenie, czy dziadek dotarł do celu
        distance_to_target = math.hypot(grandpa_pos[0] - grandpa_target[0], grandpa_pos[1] - grandpa_target[1])
        if distance_to_target < 10:  # Jeśli odległość jest mała, resetuj cel
            grandpa_target = random_target()

        grandpa_pos = move_toward(grandpa_pos, grandpa_target, grandpa_speed)

    # Teleportacja dziadka po przejściu przez krawędź
    grandpa_pos = teleport_if_necessary(grandpa_pos)

    # Rysuj dziadka
    pygame.draw.circle(screen, RED, (int(grandpa_pos[0]), int(grandpa_pos[1])), 10)

    # Policjant reaguje na obecność dziadka poza strefą
    if policeman_chasing:
        policeman_pos = move_toward(policeman_pos, grandpa_pos, policeman_speed)  # Policjant goni dziadka
    else:
        # Policjant oddala się od strefy, jeśli dziadek ją opuścił
        for radius in [50, 100, 150]:
            move_away = move_away_from_zone(policeman_pos, radius, policeman_speed)
            if move_away != [0, 0]:
                policeman_pos[0] += move_away[0]
                policeman_pos[1] += move_away[1]
                break

    # Policjant nie zbliża się do strefy (kolorowe okręgi)
    distance_to_blue_zone = math.hypot(policeman_pos[0] - center[0], policeman_pos[1] - center[1])
    if distance_to_blue_zone <= 150:
        policeman_pos = move_away_from_zone(policeman_pos, 150, policeman_speed)  # Zatrzymuje się i oddala

    # Rysowanie policjanta
    pygame.draw.circle(screen, (0, 0, 0), (int(policeman_pos[0]), int(policeman_pos[1])), 10)

    # Wyświetlanie statusu
    if time_in_house is not None:
        status = "Dziadek w domu!"
    else:
        status = "Policjant goni dziadka" if policeman_chasing else "Policjant oddala się od strefy"

    text = font.render(status, True, BLACK)
    screen.blit(text, (20, 20))

    draw_stars(screen, 1)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
