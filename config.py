"""Configuración central del coche autónomo con NEAT.

Consolida todos los parámetros que antes estaban dispersos como variables
globales en ``car_train_advanced_rox.py``. Un único lugar para leer y ajustar,
sin tocar la lógica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ------------------------------------------------------------- Pantalla
    screen_width: int = 1280
    screen_height: int = 720
    fps: int = 120

    # --------------------------------------------------------------- Coche
    available_speeds: Tuple[int, ...] = (2, 3, 4)
    rotation_speed: float = 5.0
    car_image: str = "images/coche.png"

    # ------------------------------------------------------------- Sensores
    sensor_max_distance: int = 350
    sensor_angles: Tuple[int, ...] = (-90, -45, 0, 45, 90)
    sensor_noise: bool = True
    sensor_noise_pct: float = 0.10

    # -------------------------------------------------------- Entrenamiento
    max_generations: int = 20
    max_frames_per_track: int = 6000
    direction_threshold: float = 0.3
    speed_threshold: float = 0.3

    # Pistas de entrenamiento (rutas relativas a la raíz del repo).
    tracks: Tuple[str, ...] = (
        "images/track1.png",
        "images/track2.png",
        "images/track3.png",
        "images/track4.png",
        "images/track5.png",
        "images/track6.png",
        "images/track7.png",
        "images/track8.webp",
        "images/track9.webp",
    )

    # -------------------------------------------------------------- Fitness
    rapid_crash_frame_threshold: int = 500
    rapid_crash_penalty: float = -1000.0
    survival_bonus: float = 500.0

    # -------------------------------------------------------------- Visual
    show_stats: bool = True
    show_sensors_debug: bool = False

    # --------------------------------------------------------------- Colores
    grass_color: Tuple[int, ...] = (34, 139, 34, 255)
    track_pixel_color: Tuple[int, ...] = (80, 80, 80, 255)
    start_line_color: Tuple[int, ...] = (255, 0, 0, 255)
    finish_color: Tuple[int, ...] = (140, 82, 255)

    # ---------------------------------------------------------------- Rutas
    neat_config_path: str = "config-feedforward.txt"
    models_dir: str = "models"
    # Genoma por defecto para la demo.
    demo_genome: str = "models/campeon-ultima-generacion.pkl"
    demo_track: str = "images/track6.png"

    @property
    def max_speed(self) -> int:
        return max(self.available_speeds)
