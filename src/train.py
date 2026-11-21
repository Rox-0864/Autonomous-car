"""Entrenamiento del coche autónomo con NEAT.

Refactoriza ``car_train_advanced_rox.py`` en una clase ``Trainer`` con estado
encapsulado (antes eran variables globales y código muerto).
"""

from __future__ import annotations

import copy
import os
import pickle

import neat
import pygame

from src.car import Car
from src.fitness import calculate_fitness
from src.track import load_tracks


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.generation = -1
        self.best_genome = None
        self.best_fitness = 0.0
        self.best_gen_fitness = 0.0

        pygame.init()
        self.screen = pygame.display.set_mode((cfg.screen_width, cfg.screen_height))
        pygame.display.set_caption("Coche Autónomo con NEAT")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.font_small = pygame.font.SysFont("Arial", 20)

        self.tracks = load_tracks(cfg.tracks, cfg.start_line_color, cfg.grass_color)
        self.first_pos = list(self.tracks.values())[0]["pos_inicial"]

    # ------------------------------------------------------------------ eval
    def eval_genomes(self, genomes, neat_config):
        self.generation += 1
        self.best_gen_fitness = 0.0

        for i, (_, genome) in enumerate(genomes):
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, neat_config)

            for track_path, data in self.tracks.items():
                car = Car(self.cfg)
                car.reset_for_track(data["pos_inicial"], data["angulo_inicial"])
                track_surface = data["surface"]
                frame = 0

                while car.alive and frame < self.cfg.max_frames_per_track:
                    frame += 1
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            raise SystemExit()
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                            self.cfg.show_sensors_debug = not self.cfg.show_sensors_debug

                    inputs = car.build_inputs(track_surface)
                    output = net.activate(inputs)

                    if output[0] > self.cfg.direction_threshold:
                        car.rotate(left=True)
                    elif output[0] < -self.cfg.direction_threshold:
                        car.rotate(right=True)

                    if output[1] < -self.cfg.speed_threshold:
                        car.set_speed(0)
                    elif output[1] > self.cfg.speed_threshold:
                        car.set_speed(2)
                    else:
                        car.set_speed(1)

                    car.move()
                    car.check_collision(track_surface)

                    if self.cfg.show_stats:
                        self._render(car, track_surface, track_path, i + 1, len(genomes), genome)

                alive_at_end = car.alive
                progress = car.net_progress()
                genome.fitness += calculate_fitness(
                    car.distance_traveled, car.time_elapsed, progress,
                    alive_at_end, self.cfg.max_frames_per_track, self.cfg,
                )

            self.best_gen_fitness = max(self.best_gen_fitness, genome.fitness)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = copy.deepcopy(genome)

    def _render(self, car, track_surface, track_path, idx, total, genome):
        self.screen.blit(track_surface, (0, 0))
        car.draw(self.screen, draw_sensors=self.cfg.show_sensors_debug)

        y = 10
        lines = [
            f"Generación: {self.generation}",
            f"Evaluando genoma: {idx}/{total}",
            f"Pista: {track_path}",
            f"Tiempo: {car.time_elapsed / self.cfg.fps:.1f}s | Velocidad: {car.speed}",
            f"Fitness parcial: {int(genome.fitness)}",
            f"Mejor fitness generación: {int(self.best_gen_fitness)}",
            f"Récord histórico: {int(self.best_fitness)}",
        ]
        for line in lines:
            self.screen.blit(self.font_small.render(line, True, (255, 255, 0)), (10, y))
            y += 25

        tip = "Presiona S para mostrar/ocultar sensores"
        self.screen.blit(self.font_small.render(tip, True, (255, 255, 255)),
                         (10, self.cfg.screen_height - 30))
        pygame.display.flip()
        self.clock.tick(self.cfg.fps)

    # ------------------------------------------------------------------ run
    def run(self, neat_config_path: str):
        neat_config = neat.config.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, neat_config_path,
        )
        pop = neat.Population(neat_config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        print(f"🚀 Entrenando en {len(self.tracks)} pistas, {self.cfg.max_generations} generaciones.")
        print(f"⚡ Velocidades: {self.cfg.available_speeds}")

        try:
            winner = pop.run(self.eval_genomes, self.cfg.max_generations)
        except neat.CompleteExtinctionException:
            print("\n⚠️ Extinción completa. Guardando el mejor genoma histórico.")
            winner = self.best_genome or stats.best_genome()

        self._save(winner)
        return winner

    def _save(self, winner):
        os.makedirs(self.cfg.models_dir, exist_ok=True)
        if self.best_genome:
            with open(os.path.join(self.cfg.models_dir, "campeon-historico.pkl"), "wb") as f:
                pickle.dump(self.best_genome, f)
        if winner:
            with open(os.path.join(self.cfg.models_dir, "campeon-ultima-generacion.pkl"), "wb") as f:
                pickle.dump(winner, f)
        print(f"💾 Modelos guardados en '{self.cfg.models_dir}/'.")
