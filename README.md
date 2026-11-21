# Autonomous Car — Coche Autónomo con NEAT 🚗

Coche que aprende a conducir solo usando **NEAT** (NeuroEvolution of Augmenting
Topologies) y **Pygame**. El coche lee 5 sensores de distancia y su velocidad,
y una red neuronal evolutiva decide hacia dónde girar y a qué velocidad ir.

Este repositorio **refactoriza y corrige** el proyecto original (scripts sueltos
con código duplicado y un bug en el fitness). Los errores corregidos están en la
**[guía de estudio](GUIA_DE_ESTUDIO.md)**.

---

## Características

- **Evolución con NEAT**: población de redes neuronales que mejoran por generación.
- **5 sensores** de distancia + control de **velocidad variable** (3 niveles).
- **Entrenamiento multi-pista** (anti-overfitting): cada genoma se evalúa en todas
  las pistas y su fitness es la suma.
- **Ruido en sensores** para robustez.
- Tres modos: **entrenar**, **demo** (probar el mejor genoma) y **manual**.
- Configuración centralizada en `config.py`.

---

## Estructura

```
Autonomous-car/
├── main.py                  # CLI: train / demo / manual
├── config.py                # TODA la configuración
├── config-feedforward.txt   # configuración de NEAT
├── src/
│   ├── car.py               # clase Car (única, antes duplicada)
│   ├── track.py             # carga de pistas + detección de inicio
│   ├── fitness.py           # función de fitness (bug corregido)
│   ├── train.py             # entrenamiento (Trainer)
│   ├── demo.py              # demo del mejor genoma
│   └── manual.py            # control manual
├── images/
│   ├── coche.png            # sprite del coche
│   └── (pistas: track1..track9)
└── models/                  # genomas entrenados (.pkl, no versionados)
```

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

> ⚠️ **Importante:** usá `neat-python==0.92` (ya pineado en `requirements.txt`).
> La versión `2.x` cambió el formato del archivo de configuración NEAT y **rompe**
> `config-feedforward.txt`.

## Pistas

Las pistas de entrenamiento (`images/track1.png` … `images/track9.webp`) **no
están incluidas** en este repositorio (son imágenes del tutorial original).
Necesitás colocarlas en `images/` con esos nombres.

Cada pista debe tener:
- Fondo **verde** (césped) y camino **gris**.
- Una línea **roja** que marca la salida (el coche detecta el inicio).
- (Opcional) un bloque **morado** `(140, 82, 255)` como meta para la demo.

## Uso

```bash
# Entrenar el modelo (genera los .pkl en models/)
python main.py train

# Demo del mejor genoma
python main.py demo

# Demo con un genoma/pista específicos
python main.py demo --genome models/ganador.pkl --track images/track3.png

# Control manual (flechas; ESPACIO para reiniciar)
python main.py manual
```

---

## Cómo funciona

1. **NEAT** evoluciona una población de redes neuronales.
2. Cada red recibe **6 entradas**: 5 distancias de sensor (normalizadas) + la
   velocidad (normalizada).
3. Produce **2 salidas**: dirección (girar) y velocidad (lenta/media/rápida).
4. El **fitness** premia recorrer distancia en menos tiempo, sobrevivir, y
   castiga chocar rápido.
5. Tras `max_generations`, se guarda el **campeón histórico** y el de la
   **última generación** en `models/`.

## Configuración

Todos los parámetros están en `config.py`: tamaño de pantalla, velocidades,
ángulos de sensores, ruido, generaciones, umbrales, pistas y colores. La
configuración de NEAT (población, mutaciones, especies) está en
`config-feedforward.txt`.

---

## Guía de estudio

Los errores del proyecto original y sus correcciones están explicados en
**[GUIA_DE_ESTUDIO.md](GUIA_DE_ESTUDIO.md)**.
