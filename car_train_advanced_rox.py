import pygame
import math
import neat
import os
import pickle
import random
import copy
import time

# ==========================================
# VARIABLES MODIFICABLES PARA OPTIMIZACIÓN
# ==========================================

# --- PARÁMETROS DEL COCHE ---
VELOCIDADES_DISPONIBLES = [2, 3, 4]  
VELOCIDAD_ROTACION = 5

# --- PARÁMETROS DE SENSORES ---
DISTANCIA_MAX_SENSOR = 350
ANGULOS_SENSORES = [-90, -45, 0, 45, 90]

# --- PARÁMETROS DE ENTRENAMIENTO ---
MAX_GENERACIONES = 20
UMBRAL_DECISION_DIRECCION = 0.3
UMBRAL_DECISION_VELOCIDAD = 0.3

# --- MEJORAS DE FITNESS Y ANTI-ESTANCAMIENTO ---
UMBRAL_TIEMPO_CHOQUE_RAPIDO = 500  # En frames (aprox. 4.16 segundos a 120 FPS)
PENALIZACION_CHOQUE_RAPIDO = -1000 # Castigo severo por morir rápido

# --- MEJORAS ANTI-OVERFITTING ---
PISTAS_ENTRENAMIENTO = [
    'images/track1.png',
    'images/track2.png',
    'images/track3.png',
    'images/track4.png',
    'images/track5.png',
    'images/track6.png',
    'images/track7.png', 
    'images/track8.webp', 
    'images/track9.webp' 
    ]
    
RUIDO_SENSORES = True
RUIDO_PORCENTAJE = 0.10

# --- MEJORAS VISUALES (SIN AFECTAR LÓGICA) ---
MOSTRAR_ESTADISTICAS = True
MOSTRAR_SENSORES_DEBUG = False

# --- PARÁMETROS VISUALES ---
FPS = 120

# ==========================================
# VARIABLES FIJAS (NO MODIFICAR)
# ==========================================

ANCHO_PANTALLA = 1280
ALTO_PANTALLA = 720
GENERACION = -1

COLOR_FONDO_CESPED = (34, 139, 34, 255)
COLOR_PISTA_PIXEL = (80, 80, 80, 255)
COLOR_SALIDA = (255, 0, 0, 255)

mejor_genoma_historico = None
mejor_fitness_historico = 0
mejor_tiempo_historico = float('inf')

pygame.init()
PANTALLA = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
pygame.display.set_caption("Coche Autónomo con NEAT - Velocidad Variable y Tiempo Optimizado")
reloj = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 30)
font_pequeña = pygame.font.SysFont("Arial", 20)

try:
    COCHE_IMG = pygame.image.load('images/coche.png').convert_alpha()
except pygame.error as e:
    print(f"Error al cargar 'images/coche.png': {e}"); pygame.quit(); exit()

estadisticas_generacion = {
    'mejor_fitness_historico': 0,
    'mejor_fitness_actual': 0,
    'mejor_tiempo_historico': float('inf')
}

def encontrar_punto_de_inicio(pista_mask):
    """
    Encuentra el punto central y el ángulo de inicio en cualquier pista.
    """
    puntos_rojos = []
    for x in range(pista_mask.get_width()):
        for y in range(pista_mask.get_height()):
            if pista_mask.get_at((x, y)) == COLOR_SALIDA:
                puntos_rojos.append((x, y))
    
    if not puntos_rojos:
        raise ValueError("No se encontró la línea de salida (roja) en la pista.")

    x_avg = sum(p[0] for p in puntos_rojos) / len(puntos_rojos)
    y_avg = sum(p[1] for p in puntos_rojos) / len(puntos_rojos)

    punto1, punto2 = None, None
    max_dist_sq = -1
    for i in range(len(puntos_rojos)):
        for j in range(i + 1, len(puntos_rojos)):
            p1 = puntos_rojos[i]
            p2 = puntos_rojos[j]
            dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                punto1, punto2 = p1, p2

    angulo_linea_rad = math.atan2(punto2[1] - punto1[1], punto2[0] - punto1[0])
    angulo_linea_deg = math.degrees(angulo_linea_rad)
    
    angulo_inicio_opcion_A = angulo_linea_deg + 90
    angulo_inicio_opcion_B = angulo_linea_deg - 90

    distancia_prueba = 30
    angulo_inicio_math = None

    for angulo_prueba in [angulo_inicio_opcion_A, angulo_inicio_opcion_B]:
        rad_prueba = math.radians(angulo_prueba)
        punto_prueba_x = int(x_avg + distancia_prueba * math.cos(rad_prueba))
        punto_prueba_y = int(y_avg + distancia_prueba * math.sin(rad_prueba))
        
        try:
            color_pixel = pista_mask.get_at((punto_prueba_x, punto_prueba_y))
            if color_pixel != COLOR_FONDO_CESPED:
                angulo_inicio_math = angulo_prueba
                break
        except IndexError:
            continue

    if angulo_inicio_math is None:
        raise ValueError("No se pudo determinar la dirección de avance desde la línea de salida.")

    angulo_final_pygame = (-angulo_inicio_math + 90) % 360
    return [x_avg, y_avg], angulo_final_pygame

PISTAS_DATA = {}
for pista_archivo in PISTAS_ENTRENAMIENTO:
    try:
        pista_surface = pygame.image.load(pista_archivo).convert()
        pos, ang = encontrar_punto_de_inicio(pista_surface)
        PISTAS_DATA[pista_archivo] = {
            'surface': pista_surface,
            'pos_inicial': pos,
            'angulo_inicial': ang
        }
    except Exception as e:
        print(f"Error fatal cargando la pista {pista_archivo}: {e}")
        pygame.quit(); exit()
POSICION_INICIAL_GLOBAL, ANGULO_INICIAL_GLOBAL = list(PISTAS_DATA.values())[0]['pos_inicial'], list(PISTAS_DATA.values())[0]['angulo_inicial']

class Coche:
    def __init__(self):
        self.sprite_original = COCHE_IMG
        self.posicion = POSICION_INICIAL_GLOBAL[:]
        self.angulo = ANGULO_INICIAL_GLOBAL
        self.velocidad = VELOCIDADES_DISPONIBLES[0]
        self.vivo = True
        self.distancia_recorrida = 0
        self.tiempo_transcurrido = 0
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angulo, 1)
        self.rect = self.sprite.get_rect(center=self.posicion)
        self.sensores, self.distancias_sensores = [], [0] * len(ANGULOS_SENSORES)
        # CORRECCIÓN: Inicialización de posicion_inicial
        self.posicion_inicial = self.posicion[:]

    def reset_para_nueva_pista(self, posicion_inicial, angulo_inicial):
        self.posicion = posicion_inicial[:]
        self.angulo = angulo_inicial
        self.velocidad = VELOCIDADES_DISPONIBLES[0]
        self.vivo = True
        self.distancia_recorrida = 0
        self.tiempo_transcurrido = 0
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angulo, 1)
        self.rect = self.sprite.get_rect(center=self.posicion)
        # CORRECCIÓN: Reset de posicion_inicial
        self.posicion_inicial = posicion_inicial[:]

    def dibujar(self, pantalla, es_mejor=False):
        pantalla.blit(self.sprite, self.rect)
        if MOSTRAR_SENSORES_DEBUG and es_mejor:
            self.dibujar_sensores(pantalla)

    def dibujar_sensores(self, pantalla):
        for i, distancia in enumerate(self.distancias_sensores):
            ang_sensor = self.angulo + ANGULOS_SENSORES[i]
            x_fin = self.posicion[0] - math.sin(math.radians(ang_sensor)) * distancia
            y_fin = self.posicion[1] - math.cos(math.radians(ang_sensor)) * distancia
            intensidad = min(255, int((distancia / DISTANCIA_MAX_SENSOR) * 255))
            color = (255 - intensidad, intensidad, 0)
            pygame.draw.line(pantalla, color, self.posicion, (x_fin, y_fin), 2)

    def mover(self):
        if not self.vivo: return
        self.distancia_recorrida += self.velocidad
        self.tiempo_transcurrido += 1
        rads = math.radians(self.angulo)
        self.posicion[1] -= math.cos(rads) * self.velocidad
        self.posicion[0] -= math.sin(rads) * self.velocidad
        self.rect.center = self.posicion

    def rotar(self, izquierda=False, derecha=False):
        if not self.vivo: return
        if izquierda: self.angulo = (self.angulo + VELOCIDAD_ROTACION) % 360
        elif derecha: self.angulo = (self.angulo - VELOCIDAD_ROTACION) % 360
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angulo, 1)
        self.rect = self.sprite.get_rect(center=self.posicion)

    def cambiar_velocidad(self, velocidad_index):
        if not self.vivo: return
        if 0 <= velocidad_index < len(VELOCIDADES_DISPONIBLES):
            self.velocidad = VELOCIDADES_DISPONIBLES[velocidad_index]
        
    def comprobar_colision(self, pista_mask):
        if not self.vivo: return
        ancho, alto = self.sprite_original.get_size()
        cx, cy = self.posicion
        rad = math.radians(self.angulo)
        cos, sin = math.cos(rad), math.sin(rad)
        hw, hh = ancho / 2, alto / 2
        puntos = [(cx + hw*cos - hh*sin, cy - hw*sin - hh*cos), (cx - hw*cos - hh*sin, cy + hw*sin - hh*cos),
                  (cx - hw*cos + hh*sin, cy + hw*sin + hh*cos), (cx + hw*cos + hh*sin, cy - hw*sin + hh*cos)]
        for px, py in puntos:
            if not (0 <= px < ANCHO_PANTALLA and 0 <= py < ALTO_PANTALLA) or pista_mask.get_at((int(px), int(py))) == COLOR_FONDO_CESPED:
                self.vivo = False
                break
    
    def actualizar_sensores(self, pista_mask):
        if not self.vivo: 
            # Devolver lista de 5 ceros si no está vivo para consistencia de input
            return [0] * len(ANGULOS_SENSORES) 
            
        self.sensores.clear(); self.distancias_sensores = [0] * len(ANGULOS_SENSORES)
        for i, ang_sensor in enumerate(ANGULOS_SENSORES):
            longitud, ang_total = 0, self.angulo + ang_sensor
            x_ini, y_ini = int(self.posicion[0]), int(self.posicion[1])
            x_fin, y_fin = x_ini, y_ini
            while longitud < DISTANCIA_MAX_SENSOR:
                longitud += 1
                x_fin = int(x_ini - math.sin(math.radians(ang_total)) * longitud)
                y_fin = int(y_ini - math.cos(math.radians(ang_total)) * longitud)
                if not (0 <= x_fin < ANCHO_PANTALLA and 0 <= y_fin < ALTO_PANTALLA) or pista_mask.get_at((x_fin, y_fin)) == COLOR_FONDO_CESPED:
                    break
            self.distancias_sensores[i] = longitud
        
        if RUIDO_SENSORES:
            for i in range(len(self.distancias_sensores)):
                ruido = random.uniform(-RUIDO_PORCENTAJE, RUIDO_PORCENTAJE) * self.distancias_sensores[i]
                self.distancias_sensores[i] = max(0, self.distancias_sensores[i] + ruido)
        
        # Devolver distancias normalizadas
        return [d / DISTANCIA_MAX_SENSOR for d in self.distancias_sensores]
    
    def calcular_progreso_neto(self):
        """Calcula la distancia euclidiana entre la posición inicial y la actual."""
        x_ini, y_ini = self.posicion_inicial
        x_act, y_act = self.posicion
        
        progreso_neto = math.sqrt((x_act - x_ini)**2 + (y_act - y_ini)**2)
        return progreso_neto           

def calcular_fitness(distancia_recorrida, tiempo, progreso_neto, esta_vivo_al_final, max_frames):
    global UMBRAL_TIEMPO_CHOQUE_RAPIDO, PENALIZACION_CHOQUE_RAPIDO

    if tiempo <= 0:
        return 0

    # 1. Componente de Distancia y Eficiencia Temporal
    # Reducimos el multiplicador de velocidad/tiempo para que la supervivencia sea más importante.
    bonus_tiempo = (distancia_recorrida / tiempo) * 20 # REDUCIR de 50 a 20 o 30
    
    # 2. Componente de Progreso Neto (Progreso es clave, pero menos agresivo)
    # Reducimos el peso del progreso neto.
    fitness = (distancia_recorrida * 0.5) + (progreso_neto * 1.0) + bonus_tiempo # REDUCIR de 1.5 a 1.0
    
    # 3. COMPONENTE ANTI-ESTANCAMIENTO/BONUS DE SUPERVIVENCIA
    
    # Bonificación por supervivencia (llegar al tiempo máximo)
    if esta_vivo_al_final or tiempo == max_frames:
        fitness += 500 # AUMENTAR ESTE BONUS (de 50 a 500) para premiar la supervivencia
    
    # Penalización por choque rápido (mantener)
    elif not esta_vivo_al_final and tiempo < UMBRAL_TIEMPO_CHOQUE_RAPIDO:
         fitness += PENALIZACION_CHOQUE_RAPIDO

    return max(0, fitness)
    
    # 3. COMPONENTE ANTI-ESTANCAMIENTO: Penalización por choque rápido
    if not esta_vivo_al_final and tiempo < UMBRAL_TIEMPO_CHOQUE_RAPIDO:
        fitness += PENALIZACION_CHOQUE_RAPIDO
    
    # Bonificación por supervivencia (llegar al tiempo máximo)
    elif esta_vivo_al_final or tiempo == max_frames:
        fitness += 50

    return max(0, fitness)
    
def eval_genomes(genomes, config):
    global GENERACION, estadisticas_generacion
    global mejor_genoma_historico, mejor_fitness_historico, mejor_tiempo_historico
    GENERACION += 1
    estadisticas_generacion['mejor_fitness_actual'] = 0

    max_vel = max(VELOCIDADES_DISPONIBLES) # Para normalizar velocidad

    # Bucle principal: Evaluar cada genoma
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Bucle secundario: Probar este genoma en CADA PISTA
        for pista_archivo, datos_pista in PISTAS_DATA.items():
            coche = Coche()
            coche.reset_para_nueva_pista(datos_pista['pos_inicial'], datos_pista['angulo_inicial'])
            pista_actual_surface = datos_pista['surface']
            
            tiempo_maximo_frames = 6000
            frame_actual = 0

            # Simulación para este coche en esta pista
            while coche.vivo and frame_actual < tiempo_maximo_frames:
                frame_actual += 1
                for evento in pygame.event.get():
                    if evento.type == pygame.QUIT:
                        pygame.quit(); exit()
                    if evento.type == pygame.KEYDOWN and evento.key == pygame.K_s:
                        global MOSTRAR_SENSORES_DEBUG
                        MOSTRAR_SENSORES_DEBUG = not MOSTRAR_SENSORES_DEBUG

                # 1. PREPARAR LAS 6 ENTRADAS DE LA RED (5 sensores normalizados + velocidad normalizada)
                coche_inputs_normalizadas = coche.actualizar_sensores(pista_actual_surface)
                velocidad_normalizada = coche.velocidad / max_vel
                input_data = tuple(coche_inputs_normalizadas) + (velocidad_normalizada,)
                
                # 2. ACTIVAR LA RED NEURONAL
                output = net.activate(input_data)
                
                # 3. DECISIÓN DE DIRECCIÓN
                if output[0] > UMBRAL_DECISION_DIRECCION: 
                    coche.rotar(izquierda=True)
                elif output[0] < -UMBRAL_DECISION_DIRECCION: 
                    coche.rotar(derecha=True)
                
                # 4. DECISIÓN DE VELOCIDAD
                if output[1] < -UMBRAL_DECISION_VELOCIDAD:
                    coche.cambiar_velocidad(0)  # Velocidad lenta
                elif output[1] > UMBRAL_DECISION_VELOCIDAD:
                    coche.cambiar_velocidad(2)  # Velocidad rápida
                else:
                    coche.cambiar_velocidad(1)  # Velocidad media
                
                # 5. MOVER y COMPROBAR COLISIÓN
                coche.mover()
                coche.comprobar_colision(pista_actual_surface)

                # --- Visualización de la simulación individual ---
                if MOSTRAR_ESTADISTICAS:
                    # NOTA: En la visualización temporal solo se puede calcular con la info disponible
                    progreso_neto_temp = coche.calcular_progreso_neto()
                    fitness_temporal = calcular_fitness(
                        coche.distancia_recorrida, 
                        coche.tiempo_transcurrido, 
                        progreso_neto_temp, 
                        coche.vivo, # si esta vivo en este frame
                        tiempo_maximo_frames
                    )
                    
                    PANTALLA.blit(pista_actual_surface, (0, 0))
                    coche.dibujar(PANTALLA, es_mejor=True)
                    mostrar_estadisticas_evaluacion(i + 1, len(genomes), pista_archivo, 
                                                  genome.fitness + fitness_temporal, 
                                                  coche.tiempo_transcurrido, coche.velocidad)
                    pygame.display.flip()
                    reloj.tick(FPS)

            # --- FIN DE LA SIMULACIÓN DE LA PISTA ---
            
            # 1. Determinamos el estado final para el cálculo de FITNESS
            esta_vivo_al_final_simulacion = coche.vivo 
            progreso_neto = coche.calcular_progreso_neto() 

            # 2. ACUMULAR FITNESS
            fitness_pista = calcular_fitness(
                coche.distancia_recorrida, 
                coche.tiempo_transcurrido, 
                progreso_neto,                          
                esta_vivo_al_final_simulacion,           
                tiempo_maximo_frames                     
            )
            genome.fitness += fitness_pista

            # 3. BONUS por Récord Histórico de Tiempo (Si la pista fuera de una sola vuelta)
            # Este bonus se debe aplicar con mucha cautela, o solo si el coche completó una vuelta.
            # Lo mantengo aquí por ahora, pero considera quitarlo si no hay detección de "lap complete".
            if coche.tiempo_transcurrido < tiempo_maximo_frames and coche.vivo:
                if coche.tiempo_transcurrido < mejor_tiempo_historico:
                    genome.fitness += 5000 
                    mejor_tiempo_historico = coche.tiempo_transcurrido

        # Actualizar estadísticas globales después de que un genoma ha sido probado en todas las pistas
        if genome.fitness > estadisticas_generacion['mejor_fitness_actual']:
            estadisticas_generacion['mejor_fitness_actual'] = genome.fitness
        if genome.fitness > estadisticas_generacion['mejor_fitness_historico']:
            estadisticas_generacion['mejor_fitness_historico'] = genome.fitness

        if genome.fitness > mejor_fitness_historico:
            mejor_fitness_historico = genome.fitness
            mejor_genoma_historico = copy.deepcopy(genome)

# El resto del código (mostrar_estadisticas_evaluacion y run) estaba correcto en su indentación.
# ... (código para mostrar_estadisticas_evaluacion) ...
# ... (código para run) ...

def mostrar_estadisticas_evaluacion(genoma_actual, total_genomas, pista_actual, fitness_parcial, tiempo_actual, velocidad_actual):
    # ... (código existente) ...
    y_offset = 10
    texto_gen = font.render(f"Generación: {GENERACION}", True, (255, 255, 255))
    PANTALLA.blit(texto_gen, (10, y_offset)); y_offset += 35

    texto_eval = font_pequeña.render(f"Evaluando Genoma: {genoma_actual}/{total_genomas}", True, (255, 255, 255))
    PANTALLA.blit(texto_eval, (10, y_offset)); y_offset += 25

    texto_pista = font_pequeña.render(f"Probando en Pista: {pista_actual}", True, (255, 255, 255))
    PANTALLA.blit(texto_pista, (10, y_offset)); y_offset += 25
    
    # <<< CAMBIO: Mostrar tiempo y velocidad actual >>>
    tiempo_segundos = tiempo_actual / FPS
    texto_tiempo = font_pequeña.render(f"Tiempo: {tiempo_segundos:.1f}s | Velocidad: {velocidad_actual}", True, (255, 255, 0))
    PANTALLA.blit(texto_tiempo, (10, y_offset)); y_offset += 25
    
    texto_parcial = font_pequeña.render(f"Fitness del Genoma: {int(fitness_parcial)}", True, (255, 255, 0))
    PANTALLA.blit(texto_parcial, (10, y_offset)); y_offset += 25
    
    texto_historico = font_pequeña.render(f"Mejor Fitness Generación: {int(estadisticas_generacion['mejor_fitness_actual'])}", True, (0, 255, 255))
    PANTALLA.blit(texto_historico, (10, y_offset)); y_offset += 25

    texto_record = font_pequeña.render(f"Récord Histórico: {int(estadisticas_generacion['mejor_fitness_historico'])}", True, (0, 255, 0))
    PANTALLA.blit(texto_record, (10, y_offset)); y_offset += 25

    # <<< CAMBIO: Mostrar récord de tiempo histórico >>>
    if mejor_tiempo_historico != float('inf'):
        tiempo_record_segundos = mejor_tiempo_historico / FPS
        texto_tiempo_record = font_pequeña.render(f"Récord Histórico Tiempo: {tiempo_record_segundos:.1f}s", True, (0, 255, 0))
        PANTALLA.blit(texto_tiempo_record, (10, y_offset)); y_offset += 25

    instruccion_txt = "Presiona S para mostrar/ocultar sensores"
    instruccion_color = (255, 255, 255) if MOSTRAR_SENSORES_DEBUG else (200, 200, 200)
    instruccion = font_pequeña.render(instruccion_txt, True, instruccion_color)
    PANTALLA.blit(instruccion, (10, ALTO_PANTALLA - 30))

def run(config_path):
    # ... (código existente) ...
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    print("🚀 Iniciando entrenamiento con método de GENERALIZACIÓN REAL + Velocidad Variable")
    print(f"📊 Cada genoma será evaluado en TODAS las pistas: {PISTAS_ENTRENAMIENTO}")
    print(f"🏆 El fitness será la SUMA del rendimiento (distancia/tiempo) en todas las pistas.")
    print(f"⚡ Velocidades disponibles: {VELOCIDADES_DISPONIBLES}")
    print(f"🎯 Objetivo: Máxima distancia en MENOR tiempo")
    print(f"📊 Presiona 'S' durante la ejecución para mostrar/ocultar sensores")
    
    try:
        ganador = pop.run(eval_genomes, MAX_GENERACIONES)
        
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"🏆 Mejor fitness AGREGADO alcanzado: {estadisticas_generacion['mejor_fitness_historico']}")
        print(f"🧬 Mejor genoma:\n{ganador}")
        
        with open('campeon-historico.pkl', 'wb') as f:
            pickle.dump(mejor_genoma_historico, f)
            print("💾 Campeón histórico guardado en 'campeon-historico.pkl'")
            
        with open('campeon-ultima-generacion.pkl', 'wb') as f:
            pickle.dump(ganador, f)
            print("💾 Campeón última generación guardado en 'campeon-final.pkl'")

    except neat.CompleteExtinctionException:
        print("\n" + "="*60 + "\n⚠️  EXTINCIÓN COMPLETA\n" + "="*60)
        ganador_estancado = stats.best_genome()
        if ganador_estancado:
            print(f"🏆 Mejor fitness antes de extinción: {ganador_estancado.fitness}")
            print(f"🧬 Genoma: {ganador_estancado}")
            with open('ganador-extincion.pkl', 'wb') as f:
                pickle.dump(ganador_estancado, f)
                print("💾 Genoma guardado en 'ganador-extincion.pkl'")
        else:
            print("No se pudo recuperar ningún genoma antes de la extinción.")

        if mejor_genoma_historico:
            with open('campeon-historico-extincion.pkl', 'wb') as f:
                pickle.dump(mejor_genoma_historico, f)
                print("💾 Campeón histórico guardado en 'campeon-historico-extincion.pkl'")


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-advanced-rox.txt')
    run(config_path)