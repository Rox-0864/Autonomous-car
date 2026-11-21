"""Punto de entrada del proyecto.

Uso:
    python main.py train                 # entrena el modelo (NEAT)
    python main.py demo                  # demo del mejor genoma
    python main.py demo --track images/track3.png --genome models/ganador.pkl
    python main.py manual                # control manual con flechas
"""

from __future__ import annotations

import argparse
import os

from config import Config


def main() -> int:
    p = argparse.ArgumentParser(description="Coche autónomo con NEAT")
    p.add_argument("mode", choices=["train", "demo", "manual"])
    p.add_argument("--track", default=None, help="ruta de la pista (demo/manual)")
    p.add_argument("--genome", default=None, help="ruta del genoma .pkl (demo)")
    args = p.parse_args()

    cfg = Config()

    if args.mode == "train":
        from src.train import Trainer

        trainer = Trainer(cfg)
        trainer.run(cfg.neat_config_path)
        return 0

    if args.mode == "demo":
        from src.demo import run_demo

        genome = args.genome or cfg.demo_genome
        track = args.track or cfg.demo_track
        for path in (genome, track):
            if not os.path.exists(path):
                print(f"❌ No se encontró: {path}")
                return 1
        result = run_demo(cfg, genome, track)
        if result:
            estado = "LLEGÓ A LA META" if result["llegó_a_meta"] else ("CHOCÓ" if result["chocó"] else "DETENIDO")
            print(f"🏁 {estado} | Pista: {result['pista']} | "
                  f"Distancia: {result['distancia']} px | Tiempo: {result['tiempo_segundos']}s")
        return 0

    from src.manual import run_manual

    track = args.track or cfg.demo_track
    run_manual(cfg, track)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
