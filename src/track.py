"""Carga de pistas y detección del punto de inicio.

La pista se representa como una imagen: fondo verde (césped), camino gris y una
línea roja que marca la salida. La meta es un bloque morado (para la demo).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pygame


def find_start_point(
    track_surface,
    start_line_color: Tuple[int, ...] = (255, 0, 0, 255),
    grass_color: Tuple[int, ...] = (34, 139, 34, 255),
) -> Tuple[list, float]:
    """Encuentra el centro de la línea de salida y el ángulo de avance.

    Devuelve ``([x, y], angulo)``, o lanza ``ValueError`` si no hay línea roja.
    """
    red_pixels: List[Tuple[int, int]] = []
    for x in range(track_surface.get_width()):
        for y in range(track_surface.get_height()):
            if track_surface.get_at((x, y)) == start_line_color:
                red_pixels.append((x, y))

    if not red_pixels:
        raise ValueError("No se encontró la línea de salida (roja) en la pista.")

    x_avg = sum(p[0] for p in red_pixels) / len(red_pixels)
    y_avg = sum(p[1] for p in red_pixels) / len(red_pixels)

    # Extremos de la línea (para calcular su orientación).
    p1, p2 = None, None
    max_dist_sq = -1.0
    for i in range(len(red_pixels)):
        for j in range(i + 1, len(red_pixels)):
            a, b = red_pixels[i], red_pixels[j]
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d > max_dist_sq:
                max_dist_sq, p1, p2 = d, a, b

    line_angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

    # Dos direcciones candidatas (perpendiculares a la línea de salida).
    candidates = (line_angle + 90, line_angle - 90)
    start_angle = None
    probe = 30  # píxeles hacia delante para probar si hay camino.

    for cand in candidates:
        rad = math.radians(cand)
        px = int(x_avg + probe * math.cos(rad))
        py = int(y_avg + probe * math.sin(rad))
        try:
            if track_surface.get_at((px, py)) != grass_color:
                start_angle = cand
                break
        except IndexError:
            continue

    if start_angle is None:
        raise ValueError("No se pudo determinar la dirección de avance.")

    pygame_angle = (-start_angle + 90) % 360
    return [x_avg, y_avg], pygame_angle


def load_tracks(
    track_paths,
    start_line_color: Tuple[int, ...],
    grass_color: Tuple[int, ...],
) -> Dict[str, dict]:
    """Carga varias pistas y calcula sus puntos de inicio.

    Devuelve ``{ruta: {'surface', 'pos_inicial', 'angulo_inicial'}}``.
    """
    data: Dict[str, dict] = {}
    for path in track_paths:
        surface = pygame.image.load(path).convert()
        pos, ang = find_start_point(surface, start_line_color, grass_color)
        data[path] = {"surface": surface, "pos_inicial": pos, "angulo_inicial": ang}
    return data
