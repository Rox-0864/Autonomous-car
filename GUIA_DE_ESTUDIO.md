# Guía de estudio — errores del proyecto original y sus correcciones

Al refactorizar el coche autónomo (`car_train_advanced_rox.py` y compañía) en un
proyecto modular, encontramos los siguientes problemas. Cada uno sigue el
formato **problema → por qué es un error → corrección**.

---

## 1. Código muerto en la función de fitness

### El problema
`calcular_fitness()` en `car_train_advanced_rox.py` tenía un `return` en la
mitad, y debajo quedaba una **segunda versión** de la lógica anti-estancamiento
que **nunca se ejecutaba**:

```python
def calcular_fitness(...):
    ...
    return max(0, fitness)      # ← la función termina acá

    # 3. COMPONENTE ANTI-ESTANCAMIENTO: Penalización por choque rápido
    if not esta_vivo_al_final and tiempo < UMBRAL_TIEMPO_CHOQUE_RAPIDO:
        fitness += PENALIZACION_CHOQUE_RAPIDO   # ← código INALCANZABLE
    ...
```

### Por qué es un error
El código muerto confunde: parece que hay dos formas de calcular el fitness, y
quien lo modifica no sabe cuál es la que realmente corre. Es un "semáforo
apagado" que distrae.

### La corrección
Se dejó **una sola** versión de `calculate_fitness()` en `src/fitness.py`, sin
código muerto y con cada componente documentado.

---

## 2. Sin documentación ni dependencias

### El problema
El repositorio no tenía `README.md`, `requirements.txt` ni `.gitignore`.

### Por qué es un error
Sin README nadie sabe qué es ni cómo correrlo; sin `requirements.txt` no es
reproducible (`pygame`, `neat-python` no están declarados).

### La corrección
Se agregaron `README.md`, `requirements.txt` (pygame + neat-python) y
`.gitignore`.

---

## 3. Assets faltantes

### El problema
El código referencia `images/track1.png` … `images/track9.webp` e
`images/coche.png`, pero el repo **solo tenía `coche.png` en la raíz** (no en
`images/`). El proyecto **no podía correr** tal cual.

### Por qué es un error
Un repo que no corre no sirve. Las rutas no coincidían con la estructura real.

### La corrección
- Se movió `coche.png` a `images/`.
- Se documentó en el README que las pistas (`track1..track9`) se colocan en
  `images/` y qué colores debe tener cada pista.

---

## 4. Binarios (.pkl) versionados

### El problema
`ganador.pkl`, `campeon-historico.pkl` y `campeon-ultima-generacion.pkl` estaban
commiteados a git.

### Por qué es un error
Los modelos entrenados son artefactos generados: no deberían versionarse (cambian
con cada entrenamiento y ensucian el historial).

### La corrección
Se movieron a `models/` (ignorado por `.gitignore`) y se regeneran con
`python main.py train`.

---

## 5. Script de prueba roto (car_test.py)

### El problema
`car_test.py` importaba de un módulo que **ya no existe** (`car_train`, que fue
renombrado a `car_train_advanced_rox.py`) y usaba un archivo de configuración
inexistente (`config-feedforward.txt`).

### Por qué es un error
Código que referencia módulos fantasmas rompe al ejecutarlo. Es un leftover de
una versión anterior.

### La corrección
Se consolidó en un solo `src/demo.py` funcional (reemplaza a `car_test.py` y
`car_test_advanced_rox.py`).

---

## 6. Duplicación de la clase Coche y la detección de inicio

### El problema
`Coche` y `encontrar_punto_de_inicio()` estaban **copiadas** en
`car_train_advanced_rox.py`, `car_manual.py` y (vía import) en los scripts de
prueba.

### Por qué es un error
Tres copias de la misma lógica divergen y nadie sabe cuál es la correcta.

### La corrección
Una sola clase `Car` en `src/car.py` y una sola `find_start_point()` en
`src/track.py`, reutilizadas por train/demo/manual.

---

## 7. Nombre de archivo inconsistente

### El problema
Al guardar el campeón, el código imprimía *"guardado en 'campeon-final.pkl'"*
pero en realidad guardaba en `campeon-ultima-generacion.pkl`.

### Por qué es un error
Un mensaje que no coincide con el archivo real hace perder tiempo depurando.

### La corrección
Nombres y mensajes consistentes en `Trainer._save()`.

---

## 8. Import frágil (side effects en import)

### El problema
`car_test_advanced_rox.py` hacía `from car_train_advanced_rox import ...`, lo que
**ejecutaba todo el módulo de entrenamiento al importarlo**: `pygame.init()`,
cargar todas las pistas, etc.

### Por qué es un error
Importar un módulo no debería tener efectos secundarios pesados (abrir una
ventana, cargar assets). Hace el código frágil y lento.

### La corrección
Se separó la lógica en módulos sin side effects: `car.py`, `track.py`,
`fitness.py` no hacen nada al importarse. La inicialización de Pygame ocurre
solo dentro de `Trainer`/`run_demo`/`run_manual`, al ejecutar.

---

## 9. Bonus de "récord de tiempo" sin detección de vuelta

### El problema
El entrenamiento daba un bonus de **+5000** por "récord de tiempo", pero **sin
detección de que el coche completara una vuelta** (el propio comentario del
código lo admitía: "considérelo quitarlo si no hay detección de lap complete").

### Por qué es un error
Premiar un "récord" que no se puede medir correctamente añade ruido al fitness y
puede dirigir la evolución en la dirección equivocada.

### La corrección
Se eliminó ese bonus. El fitness queda limpio (distancia + progreso + eficiencia
temporal + bonus de supervivencia + penalización por choque).

---

## 10. Configuración dispersa en variables globales

### El problema
Todos los parámetros (velocidades, ángulos, umbrales, colores, pistas) eran
variables globales sueltas al inicio de `car_train_advanced_rox.py`.

### Por qué es un error
Difícil encontrar y ajustar un parámetro, y propenso a inconsistencia entre
scripts.

### La corrección
Se centralizó todo en `config.Config` (una `dataclass`), usado por train, demo y
manual.

---

## Resumen rápido

| # | Error | Corrección |
|---|-------|------------|
| 1 | Código muerto en el fitness | `fitness.py` limpio |
| 2 | Sin README/requirements/gitignore | documentación completa |
| 3 | Assets faltantes (`images/`) | estructura + documentación |
| 4 | `.pkl` versionados | `models/` ignorado |
| 5 | `car_test.py` roto | `src/demo.py` funcional |
| 6 | Clase `Coche` duplicada | `src/car.py` único |
| 7 | Nombre de archivo inconsistente | mensajes consistentes |
| 8 | Import con side effects | módulos sin efectos al importar |
| 9 | Bonus de récord sin detección | fitness limpio |
| 10 | Config en globales | `config.py` (dataclass) |
