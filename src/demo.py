"""Demo: ejecuta un genoma entrenado en una pista y muestra el resultado."""

from __future__ import annotations

import os
import pickle
from typing import Optional

import neat
import pygame

from src.car import Car
from src.track import find_start_point


def _check_finish(car: Car, track_surface, finish_color) -> bool:
    cx, cy = int(car.position[0]), int(car.position[1])
    if 0 <= cx < track_surface.get_width() and 0 <= cy < track_surface.get_height():
        return track_surface.get_at((cx, cy))[:3] == finish_color[:3]
    return False


def run_demo(cfg, genome_path: str, track_path: str) -> Optional[dict]:
    """Ejecuta la demo y devuelve un resumen, o None si hay error de entrada."""
    if not os.path.exists(genome_path) or not os.path.exists(track_path):
        return None

    neat_config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg.neat_config_path,
    )
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

    pygame.init()
    screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))
    pygame.display.set_caption(f"Demo — {os.path.basename(track_path)}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    track_surface = pygame.image.load(track_path).convert()
    pos, ang = find_start_point(track_surface, cfg.start_line_color, cfg.grass_color)

    car = Car(cfg)
    car.reset_for_track(pos, ang)
    reached_finish = False
    show_sensors = False

    start_ms = pygame.time.get_ticks()
    running = True
    while running and car.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_ESCAPE:
                    running = False

        inputs = car.build_inputs(track_surface)
        output = net.activate(inputs)

        if output[0] > cfg.direction_threshold:
            car.rotate(left=True)
        elif output[0] < -cfg.direction_threshold:
            car.rotate(right=True)
        if len(output) > 1:
            if output[1] < -cfg.speed_threshold:
                car.set_speed(0)
            elif output[1] > cfg.speed_threshold:
                car.set_speed(2)
            else:
                car.set_speed(1)

        car.move()
        car.check_collision(track_surface)
        if car.alive and _check_finish(car, track_surface, cfg.finish_color):
            reached_finish = True
            break

        screen.blit(track_surface, (0, 0))
        car.draw(screen, draw_sensors=show_sensors)
        elapsed = (pygame.time.get_ticks() - start_ms) / 1000
        screen.blit(font.render(f"Distancia: {int(car.distance_traveled)} px", True, (255, 255, 0)), (10, 10))
        screen.blit(font.render(f"Tiempo: {elapsed:.1f}s", True, (255, 255, 0)), (10, 35))
        pygame.display.flip()
        clock.tick(cfg.fps)

    total_ms = pygame.time.get_ticks() - start_ms
    pygame.quit()

    return {
        "pista": os.path.basename(track_path),
        "llegó_a_meta": reached_finish,
        "chocó": not car.alive,
        "distancia": int(car.distance_traveled),
        "tiempo_segundos": round(total_ms / 1000, 1),
        "fitness": genome.fitness,
    }
