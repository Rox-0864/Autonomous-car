"""El coche autónomo: posición, movimiento, sensores y colisiones.

Clase única (antes duplicada en car_train, car_manual y car_test). Todos los
parámetros vienen de ``config.Config``.
"""

from __future__ import annotations

import math
import random

import pygame


class Car:
    def __init__(self, config):
        self.cfg = config
        self.sprite_original = pygame.image.load(config.car_image).convert_alpha()
        self.position: list = [0.0, 0.0]
        self.angle: float = 0.0
        self.speed: int = config.available_speeds[0]
        self.alive: bool = True
        self.distance_traveled: float = 0.0
        self.time_elapsed: int = 0
        self.start_position: list = [0.0, 0.0]
        self.sensors: list = []
        self.sensor_distances: list = [0] * len(config.sensor_angles)
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angle, 1)
        self.rect = self.sprite.get_rect(center=self.position)

    def reset_for_track(self, start_pos, start_angle):
        self.position = list(start_pos)
        self.angle = start_angle
        self.speed = self.cfg.available_speeds[0]
        self.alive = True
        self.distance_traveled = 0.0
        self.time_elapsed = 0
        self.start_position = list(start_pos)
        self._refresh_sprite()

    def _refresh_sprite(self):
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angle, 1)
        self.rect = self.sprite.get_rect(center=self.position)

    def draw(self, screen, draw_sensors=False):
        screen.blit(self.sprite, self.rect)
        if draw_sensors:
            self._draw_sensors(screen)

    def _draw_sensors(self, screen):
        for i, dist in enumerate(self.sensor_distances):
            sensor_angle = self.angle + self.cfg.sensor_angles[i]
            x_end = self.position[0] - math.sin(math.radians(sensor_angle)) * dist
            y_end = self.position[1] - math.cos(math.radians(sensor_angle)) * dist
            intensity = min(255, int((dist / self.cfg.sensor_max_distance) * 255))
            color = (255 - intensity, intensity, 0)
            pygame.draw.line(screen, color, self.position, (x_end, y_end), 2)

    def move(self):
        if not self.alive:
            return
        self.distance_traveled += self.speed
        self.time_elapsed += 1
        rads = math.radians(self.angle)
        self.position[1] -= math.cos(rads) * self.speed
        self.position[0] -= math.sin(rads) * self.speed
        self.rect.center = self.position

    def rotate(self, left=False, right=False):
        if not self.alive:
            return
        if left:
            self.angle = (self.angle + self.cfg.rotation_speed) % 360
        elif right:
            self.angle = (self.angle - self.cfg.rotation_speed) % 360
        self._refresh_sprite()

    def set_speed(self, index):
        if not self.alive:
            return
        if 0 <= index < len(self.cfg.available_speeds):
            self.speed = self.cfg.available_speeds[index]

    def check_collision(self, track_surface):
        if not self.alive:
            return
        w, h = self.sprite_original.get_size()
        cx, cy = self.position
        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        hw, hh = w / 2, h / 2
        corners = [
            (cx + hw * cos - hh * sin, cy - hw * sin - hh * cos),
            (cx - hw * cos - hh * sin, cy + hw * sin - hh * cos),
            (cx - hw * cos + hh * sin, cy + hw * sin + hh * cos),
            (cx + hw * cos + hh * sin, cy - hw * sin + hh * cos),
        ]
        for px, py in corners:
            out = not (0 <= px < self.cfg.screen_width and 0 <= py < self.cfg.screen_height)
            if out or track_surface.get_at((int(px), int(py))) == self.cfg.grass_color:
                self.alive = False
                break

    def update_sensors(self, track_surface) -> list:
        """Actualiza los sensores y devuelve las distancias normalizadas (0-1)."""
        if not self.alive:
            self.sensor_distances = [0] * len(self.cfg.sensor_angles)
            return self.sensor_distances

        self.sensors.clear()
        self.sensor_distances = [0] * len(self.cfg.sensor_angles)

        for i, sensor_angle in enumerate(self.cfg.sensor_angles):
            length = 0
            total_angle = self.angle + sensor_angle
            x_start, y_start = int(self.position[0]), int(self.position[1])
            x_end, y_end = x_start, y_start
            while length < self.cfg.sensor_max_distance:
                length += 1
                x_end = int(x_start - math.sin(math.radians(total_angle)) * length)
                y_end = int(y_start - math.cos(math.radians(total_angle)) * length)
                out = not (0 <= x_end < self.cfg.screen_width and 0 <= y_end < self.cfg.screen_height)
                if out or track_surface.get_at((x_end, y_end)) == self.cfg.grass_color:
                    break
            self.sensor_distances[i] = length

        if self.cfg.sensor_noise:
            for i in range(len(self.sensor_distances)):
                noise = random.uniform(-self.cfg.sensor_noise_pct, self.cfg.sensor_noise_pct)
                self.sensor_distances[i] = max(0, self.sensor_distances[i] * (1 + noise))

        return [d / self.cfg.sensor_max_distance for d in self.sensor_distances]

    def net_progress(self) -> float:
        """Distancia euclidiana entre el punto inicial y el actual."""
        dx = self.position[0] - self.start_position[0]
        dy = self.position[1] - self.start_position[1]
        return math.sqrt(dx * dx + dy * dy)

    def build_inputs(self, track_surface) -> tuple:
        """Construye las entradas de la red: 5 sensores normalizados + velocidad."""
        sensors = self.update_sensors(track_surface)
        speed_norm = self.speed / self.cfg.max_speed
        return tuple(sensors) + (speed_norm,)
