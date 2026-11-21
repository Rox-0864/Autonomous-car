"""Control manual del coche con el teclado (flechas)."""

from __future__ import annotations

import pygame

from src.car import Car
from src.track import find_start_point


def run_manual(cfg, track_path: str):
    pygame.init()
    screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))
    pygame.display.set_caption("Control manual")
    clock = pygame.time.Clock()

    track_surface = pygame.image.load(track_path).convert()
    pos, ang = find_start_point(track_surface, cfg.start_line_color, cfg.grass_color)

    car = Car(cfg)
    car.reset_for_track(pos, ang)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                car.reset_for_track(pos, ang)

        if car.alive:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                car.move()
            if keys[pygame.K_LEFT]:
                car.rotate(left=True)
            if keys[pygame.K_RIGHT]:
                car.rotate(right=True)

        car.update_sensors(track_surface)
        car.check_collision(track_surface)

        screen.blit(track_surface, (0, 0))
        car.draw(screen, draw_sensors=True)
        pygame.display.flip()
        clock.tick(cfg.fps)

    pygame.quit()
    print("Simulación manual terminada.")
