"""Cálculo del fitness (función objetivo de la evolución).

Corrige un bug del original: había código muerto (inalcanzable) después del
``return``. Aquí la función está limpia y todos sus componentes documentados.
"""

from __future__ import annotations


def calculate_fitness(
    distance_traveled: float,
    time_elapsed: int,
    net_progress: float,
    alive_at_end: bool,
    max_frames: int,
    cfg,
) -> float:
    """Fitness = distancia + progreso neto + bonus por eficiencia temporal.

    - Premia recorrer más distancia en menos tiempo.
    - Premia sobrevivir hasta el final (bonus de supervivencia).
    - Castiga chocar muy rápido (penalización anti-estancamiento).
    """
    if time_elapsed <= 0:
        return 0.0

    # 1. Eficiencia temporal: distancia / tiempo (cuanto antes, mejor).
    time_bonus = (distance_traveled / time_elapsed) * 20

    # 2. Fitness base: distancia recorrida + progreso neto + bonus temporal.
    fitness = (distance_traveled * 0.5) + (net_progress * 1.0) + time_bonus

    # 3. Anti-estancamiento: bonus de supervivencia o penalización por choque rápido.
    if alive_at_end or time_elapsed >= max_frames:
        fitness += cfg.survival_bonus
    elif time_elapsed < cfg.rapid_crash_frame_threshold:
        fitness += cfg.rapid_crash_penalty

    return max(0.0, fitness)
