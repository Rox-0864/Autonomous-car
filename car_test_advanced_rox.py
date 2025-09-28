import pygame
import neat
import os
import pickle
import sys

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO_GANADOR = 'campeon-ultima-generacion.pkl'  # Cambiar por el que quieras probar
PISTA_DE_PRUEBA = 'images/track6.png'
FPS_DEMO = 120

# Color de la meta (bloque morado)
COLOR_META = (140, 82, 255)  # #8c52ff

# --- IMPORTAR DE car_train_advanced ---
try:
    from car_train_advanced_rox import (
        Coche, 
        encontrar_punto_de_inicio, 
        ANCHO_PANTALLA, 
        ALTO_PANTALLA, 
        font_pequeña,
        MOSTRAR_SENSORES_DEBUG,
        VELOCIDADES_DISPONIBLES,
        UMBRAL_DECISION_DIRECCION,
        UMBRAL_DECISION_VELOCIDAD
    )
except ImportError as e:
    print(f"Error: Asegúrate de que car_train_advanced.py esté en la misma carpeta.")
    print(f"Detalles del error: {e}")
    sys.exit()

def comprobar_meta(coche, pista_surface):
    """
    Verifica si el coche ha llegado a la meta (bloque morado).
    """
    # Obtener la posición del centro del coche
    center_x, center_y = int(coche.posicion[0]), int(coche.posicion[1])
    
    # Verificar límites de la pantalla
    if (0 <= center_x < pista_surface.get_width() and 
        0 <= center_y < pista_surface.get_height()):
        
        # Obtener el color del pixel en la posición del coche
        color_pixel = pista_surface.get_at((center_x, center_y))
        
        # Verificar si es el color de la meta
        if color_pixel[:3] == COLOR_META:
            return True
    
    return False

def run_demo(config_path, genome_path, track_path):
    """
    Ejecuta una demostración del mejor genoma entrenado en una pista específica.
    """
    # Verificar que los archivos existan
    if not os.path.exists(config_path):
        print(f"Error: No se encuentra el archivo de configuración: {config_path}")
        return
    
    if not os.path.exists(genome_path):
        print(f"Error: No se encuentra el archivo del genoma: {genome_path}")
        return
    
    if not os.path.exists(track_path):
        print(f"Error: No se encuentra la pista: {track_path}")
        return

    # Cargar configuración NEAT
    try:
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_path)
    except Exception as e:
        print(f"Error cargando configuración NEAT: {e}")
        return

    # Cargar genoma ganador
    try:
        with open(genome_path, "rb") as f:
            ganador = pickle.load(f)
    except Exception as e:
        print(f"Error cargando el genoma: {e}")
        return

    # Crear red neuronal
    try:
        net = neat.nn.FeedForwardNetwork.create(ganador, config)
    except Exception as e:
        print(f"Error creando la red neuronal: {e}")
        return

    # Configurar Pygame
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
    pygame.display.set_caption(f"Prueba Campeón - {os.path.basename(track_path)}")
    reloj = pygame.time.Clock()

    # Cargar pista y encontrar inicio
    try:
        pista_surface = pygame.image.load(track_path).convert()
        pos_inicial, angulo_inicial = encontrar_punto_de_inicio(pista_surface)
    except Exception as e:
        print(f"Error cargando la pista o encontrando el punto de inicio: {e}")
        pygame.quit()
        return

    # Crear coche
    coche = Coche()
    coche.reset_para_nueva_pista(pos_inicial, angulo_inicial)
    # Agregar atributo para marcar si llegó a la meta
    coche.llego_a_meta = False
    
    # Variables para estadísticas
    tiempo_inicio = pygame.time.get_ticks()
    mostrar_sensores = False
    max_vel = max(VELOCIDADES_DISPONIBLES)

    print("\n" + "="*50)
    print("🚗 DEMOSTRACIÓN INICIADA")
    print("="*50)
    print("Controles:")
    print("  - S: Mostrar/ocultar sensores")
    print("  - ESC o cerrar ventana: Salir")
    print("  - El coche se controla automáticamente con IA")
    print("="*50)

    # Bucle principal
    corriendo = True
    frames_transcurridos = 0
    
    while corriendo and coche.vivo:
        frames_transcurridos += 1
        
        # Manejar eventos
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_s:
                    mostrar_sensores = not mostrar_sensores
                elif evento.key == pygame.K_ESCAPE:
                    corriendo = False

        # 1. Obtener entradas normalizadas
        # NOTA: coche.actualizar_sensores devuelve las 5 distancias normalizadas
        coche_inputs_normalizadas = coche.actualizar_sensores(pista_surface) 
        
        # 2. Normalizar la velocidad
        velocidad_normalizada = coche.velocidad / max_vel
        
        # 3. Combinar las 6 entradas
        input_data = tuple(coche_inputs_normalizadas) + (velocidad_normalizada,)
        
        # 4. Obtener decisión de la IA (AHORA CON 6 ENTRADAS)
        try:
            output = net.activate(input_data)
            
            # Salida 0: Dirección
            if output[0] > UMBRAL_DECISION_DIRECCION:
                coche.rotar(izquierda=True)
            elif output[0] < -UMBRAL_DECISION_DIRECCION:
                coche.rotar(derecha=True)
            
            # Salida 1: Velocidad (si está disponible)
            if len(output) > 1:
                if output[1] < -UMBRAL_DECISION_VELOCIDAD:
                    coche.cambiar_velocidad(0)  # Velocidad lenta
                elif output[1] > UMBRAL_DECISION_VELOCIDAD:
                    coche.cambiar_velocidad(2)  # Velocidad rápida
                else:
                    coche.cambiar_velocidad(1)  # Velocidad media
            
        except Exception as e:
            print(f"Error en la activación de la red neuronal: {e}")
            break
        
        # Mover y verificar colisión
        coche.mover()
        coche.comprobar_colision(pista_surface)
        
        # Verificar si llegó a la meta (solo si sigue vivo)
        if coche.vivo and comprobar_meta(coche, pista_surface):
            coche.llego_a_meta = True
            coche.vivo = False  # Marcar como no vivo para terminar el bucle

        # Dibujar todo
        pantalla.blit(pista_surface, (0, 0))
        coche.dibujar(pantalla, es_mejor=True)
        
        # Dibujar sensores si está activado
        if mostrar_sensores:
            coche.dibujar_sensores(pantalla)

        # Mostrar estadísticas
        tiempo_actual = pygame.time.get_ticks() - tiempo_inicio
        mostrar_estadisticas_demo(pantalla, coche, tiempo_actual, frames_transcurridos, ganador)

        pygame.display.flip()
        reloj.tick(FPS_DEMO)
    
    # Mostrar resultado final
    tiempo_total = pygame.time.get_ticks() - tiempo_inicio
    print("\n" + "="*50)
    print("📊 RESULTADO FINAL")
    print("="*50)
    
    if coche.llego_a_meta:
        print(f"🏆 ¡El coche llegó a la meta!")
    elif not coche.vivo:
        print(f"💥 ¡El coche chocó!")
    else:
        print(f"⏹️ Simulación finalizada por el usuario.")
    
    print(f"🏁 Pista: {os.path.basename(track_path)}")
    print(f"📏 Distancia recorrida: {int(coche.distancia_recorrida)} píxeles")
    print(f"⏱️ Tiempo total: {tiempo_total/1000:.1f} segundos")
    print(f"🎯 Velocidad promedio: {coche.distancia_recorrida/(tiempo_total/1000):.1f} píxeles/seg")
    print(f"🧬 ID del genoma: {ganador.key}")
    print(f"🏆 Fitness del genoma: {ganador.fitness}")
    print(f"🚗 Estado final: {'LLEGÓ A LA META' if coche.llego_a_meta else 'CHOCÓ'}")
    print("="*50)

    pygame.quit()

def mostrar_estadisticas_demo(pantalla, coche, tiempo_ms, frames, genoma):
    """
    Muestra estadísticas en tiempo real durante la demostración.
    """
    y_offset = 10
    tiempo_segundos = tiempo_ms / 1000
    
    # Información del genoma
    texto_id = font_pequeña.render(f"Genoma ID: {genoma.key} | Fitness: {int(genoma.fitness)}", True, (255, 255, 255))
    pantalla.blit(texto_id, (10, y_offset))
    y_offset += 25
    
    # Estadísticas del coche
    texto_distancia = font_pequeña.render(f"Distancia: {int(coche.distancia_recorrida)} px", True, (255, 255, 0))
    pantalla.blit(texto_distancia, (10, y_offset))
    y_offset += 25
    
    texto_tiempo = font_pequeña.render(f"Tiempo: {tiempo_segundos:.1f}s", True, (255, 255, 0))
    pantalla.blit(texto_tiempo, (10, y_offset))
    y_offset += 25
    
    # Velocidad actual y promedio
    if tiempo_segundos > 0:
        vel_promedio = coche.distancia_recorrida / tiempo_segundos
        texto_velocidad = font_pequeña.render(f"Velocidad: {coche.velocidad}", True, (255, 255, 0))
        pantalla.blit(texto_velocidad, (10, y_offset))
        y_offset += 25

def listar_genomas_disponibles():
    """
    Lista todos los archivos .pkl disponibles en el directorio actual.
    """
    archivos_pkl = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if not archivos_pkl:
        print("❌ No se encontraron archivos de genoma (.pkl) en el directorio actual.")
    return archivos_pkl

def main():
    # Verificar archivos necesarios
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-advanced-rox.txt')
    
    if not os.path.exists(config_path):
        print(f"❌ Error: No se encuentra el archivo de configuración: {config_path}")
        print("   Asegúrate de que 'config-feedforward-advanced-rox.txt' esté en la misma carpeta.")
        return
    
    # Listar genomas disponibles
    genomas_disponibles = listar_genomas_disponibles()
    if not genomas_disponibles:
        print("   Ejecuta primero car_train_advanced.py para entrenar un modelo.")
        return
    
    # Seleccionar genoma
    ganador_path = os.path.join(local_dir, NOMBRE_ARCHIVO_GANADOR)
    
    if not os.path.exists(ganador_path):
        print(f"❌ El archivo {NOMBRE_ARCHIVO_GANADOR} no existe.")
        if genomas_disponibles:
            primer_genoma = genomas_disponibles[0]
            print(f"🔄 Usando el primer genoma disponible: {primer_genoma}")
            ganador_path = os.path.join(local_dir, primer_genoma)
        else:
            return
    
    # Seleccionar pista
    pista_path = os.path.join(local_dir, PISTA_DE_PRUEBA)
    
    if not os.path.exists(pista_path):
        print(f"❌ La pista {PISTA_DE_PRUEBA} no existe.")
        # Buscar pistas alternativas
        pistas_disponibles = [f for f in os.listdir('.') if f.endswith('.png') and f.startswith('track')]
        if pistas_disponibles:
            pista_alternativa = pistas_disponibles[0]
            print(f"🔄 Usando pista alternativa: {pista_alternativa}")
            pista_path = os.path.join(local_dir, pista_alternativa)
        else:
            print("   No se encontraron pistas (.png) en el directorio actual.")
            return
    
    # Ejecutar demostración
    run_demo(config_path, ganador_path, pista_path)

if __name__ == '__main__':
    main()