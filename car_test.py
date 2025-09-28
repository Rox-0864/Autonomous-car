import pygame
import neat
import os
import pickle
import sys

# --- CONFIGURACIÓN ---
NOMBRE_ARCHIVO_GANADOR = 'ganador.pkl'
PISTA_DE_PRUEBA = 'images/track6.png'
FPS_DEMO = 120

# --- IMPORTAR DE car_train ---
try:
    from car_train import Coche, encontrar_punto_de_inicio, ANCHO_PANTALLA, ALTO_PANTALLA, font_pequeña
    from car_train import MOSTRAR_SENSORES_DEBUG, UMBRAL_DECISION_IA
except ImportError:
    print("Error: Asegúrate de que car_train.py esté en la misma carpeta.")
    sys.exit()

def run_demo(config_path, genome_path, track_path):
    # Cargar configuración NEAT
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    # Cargar genoma ganador
    print(f"Cargando genoma desde: {genome_path}")
    with open(genome_path, "rb") as f:
        ganador = pickle.load(f)
    print(f"Genoma cargado - ID: {ganador.key}, Fitness: {ganador.fitness}")

    # Crear red neuronal
    net = neat.nn.FeedForwardNetwork.create(ganador, config)

    # Configurar Pygame
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
    pygame.display.set_caption(f"Prueba Campeón - {os.path.basename(track_path)}")
    reloj = pygame.time.Clock()

    # Cargar pista y encontrar inicio
    pista_surface = pygame.image.load(track_path).convert()
    pos_inicial, angulo_inicial = encontrar_punto_de_inicio(pista_surface)

    # Crear coche y configurarlo manualmente
    coche = Coche()
    # Resetear posición y ángulo manualmente ya que no existe reset_para_nueva_pista()
    coche.posicion = pos_inicial[:]  # Copia la lista
    coche.angulo = angulo_inicial
    coche.vivo = True
    coche.distancia_recorrida = 0
    
    # Actualizar sprite y rect con la nueva posición/ángulo
    coche.sprite = pygame.transform.rotozoom(coche.sprite_original, coche.angulo, 1)
    coche.rect = coche.sprite.get_rect(center=coche.posicion)
    
    print(f"Coche listo en: {os.path.basename(track_path)}")
    print(f"Posición inicial: {pos_inicial}, Ángulo: {angulo_inicial}")

    # Variables para controlar la demostración
    corriendo = True
    mostrar_sensores_local = False  # Control local para sensores
    
    # Bucle principal
    while corriendo and coche.vivo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_s:
                    mostrar_sensores_local = not mostrar_sensores_local
                    print(f"Sensores: {'Activados' if mostrar_sensores_local else 'Desactivados'}")
                elif evento.key == pygame.K_ESCAPE:
                    corriendo = False

        # IA del coche
        coche.actualizar_sensores(pista_surface)
        output = net.activate(tuple(coche.distancias_sensores))
        
        # Usar el mismo umbral que en el entrenamiento
        if output[0] > UMBRAL_DECISION_IA:
            coche.rotar(izquierda=True)
        elif output[0] < -UMBRAL_DECISION_IA:
            coche.rotar(derecha=True)
        
        coche.mover()
        coche.comprobar_colision(pista_surface)

        # Dibujar
        pantalla.blit(pista_surface, (0, 0))
        coche.dibujar(pantalla, es_mejor=True)
        
        # Dibujar sensores si está activado localmente
        if mostrar_sensores_local:
            coche.dibujar_sensores(pantalla)

        # Estadísticas en pantalla
        dist_texto = font_pequeña.render(f"Distancia: {int(coche.distancia_recorrida)}", True, (255, 255, 0))
        pantalla.blit(dist_texto, (10, 10))
        
        # Información adicional
        fitness_texto = font_pequeña.render(f"Fitness del genoma: {ganador.fitness}", True, (255, 255, 255))
        pantalla.blit(fitness_texto, (10, 35))
        
        # Instrucciones
        if MOSTRAR_SENSORES_DEBUG:
            instruccion = font_pequeña.render("Presiona S para ocultar sensores", True, (255, 255, 255))
        else:
            instruccion = font_pequeña.render("Presiona S para mostrar sensores", True, (200, 200, 200))
        pantalla.blit(instruccion, (10, ALTO_PANTALLA - 30))

        pygame.display.flip()
        reloj.tick(FPS_DEMO)
    
    # Resultado final
    if not coche.vivo:
        print(f"¡Chocó! Distancia final: {int(coche.distancia_recorrida)}")
        print(f"Fitness del genoma entrenado era: {ganador.fitness}")
        if coche.distancia_recorrida < ganador.fitness * 0.8:
            print("⚠️  El rendimiento es menor al esperado. Posibles causas:")
            print("   - Diferente configuración de pista")
            print("   - Cambios en parámetros del coche")
            print("   - Diferencias en la detección de colisiones")
    else:
        print("Simulación finalizada por el usuario.")
        print(f"Distancia alcanzada: {int(coche.distancia_recorrida)}")

    pygame.quit()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    ganador_path = os.path.join(local_dir, NOMBRE_ARCHIVO_GANADOR)
    pista_path = os.path.join(local_dir, PISTA_DE_PRUEBA)
    
    # Verificar que los archivos existan
    archivos_necesarios = [
        (config_path, "Archivo de configuración NEAT"),
        (ganador_path, "Archivo del genoma ganador"),
        (pista_path, "Archivo de la pista")
    ]
    
    for archivo, descripcion in archivos_necesarios:
        if not os.path.exists(archivo):
            print(f"❌ Error: No se encontró {descripcion}: {archivo}")
            sys.exit(1)
    
    print("🚗 Iniciando demostración del coche campeón...")
    print(f"📁 Usando pista: {os.path.basename(pista_path)}")
    run_demo(config_path, ganador_path, pista_path)