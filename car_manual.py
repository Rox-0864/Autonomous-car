import pygame
import math

# --- 1. CONSTANTES Y CONFIGURACIÓN ---
ANCHO_PANTALLA = 1280
ALTO_PANTALLA = 720
FPS = 120

NOMBRE_PISTA = 'images/track1.png' 

# --- INICIALIZACIÓN DE PYGAME ---
pygame.init()
PANTALLA = pygame.display.set_mode((ANCHO_PANTALLA, ALTO_PANTALLA))
pygame.display.set_caption("Prueba de Coche")
reloj = pygame.time.Clock()

# --- CARGAR ASSETS ---
try:
    PISTA = pygame.image.load(NOMBRE_PISTA).convert()
    COCHE_IMG = pygame.image.load('images/coche.png').convert_alpha()
except pygame.error as e:
    print(f"Error al cargar una imagen: {e}"); pygame.quit(); exit()

# --- COLORES ---
COLOR_FONDO_CESPED = (34, 139, 34, 255)
COLOR_PISTA_PIXEL = (80, 80, 80, 255)
COLOR_SALIDA = (255, 0, 0, 255)

# --- FUNCIÓN DE BÚSQUEDA AUTOMÁTICA ---
def encontrar_punto_de_inicio(pista_mask):
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

    # La fórmula de conversión sigue siendo la misma y ahora debería ser correcta
    angulo_final_pygame = (-angulo_inicio_math + 90) % 360
    return [x_avg, y_avg], angulo_final_pygame

# --- 2. CLASE COCHE ---
class Coche:
    def __init__(self):
        self.sprite_original = COCHE_IMG
        try:
            self.posicion_inicial, self.angulo_inicial = encontrar_punto_de_inicio(PISTA)
        except ValueError as e:
            print(f"Error: {e}")
            self.posicion_inicial, self.angulo_inicial = [ANCHO_PANTALLA/2, ALTO_PANTALLA/2], 0
        
        self.posicion = self.posicion_inicial[:]
        self.angulo = self.angulo_inicial
        self.velocidad = 4
        self.vivo = True
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angulo, 1)
        self.rect = self.sprite.get_rect(center=self.posicion)
        self.sensores = []

    def dibujar(self, pantalla):
        pantalla.blit(self.sprite, self.rect)
        for sensor in self.sensores:
            pygame.draw.line(pantalla, (255, 255, 0), sensor[0], sensor[1], 1)
            pygame.draw.circle(pantalla, (255, 0, 0), sensor[1], 3)

    def mover(self):
        if not self.vivo: return
        rads = math.radians(self.angulo)
        self.posicion[1] -= math.cos(rads) * self.velocidad
        self.posicion[0] -= math.sin(rads) * self.velocidad
        self.rect.center = self.posicion

    def rotar(self, izquierda=False, derecha=False):
        if not self.vivo: return
        if izquierda: self.angulo = (self.angulo + 5) % 360
        elif derecha: self.angulo = (self.angulo - 5) % 360
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angulo, 1)
        self.rect = self.sprite.get_rect(center=self.posicion)
        
    def comprobar_colision(self, pista_mask):
        if not self.vivo: return
        ancho, alto = self.sprite_original.get_size()
        cx, cy = self.posicion
        rad = math.radians(self.angulo)
        cos, sin = math.cos(rad), math.sin(rad)
        hw, hh = ancho / 2, alto / 2
        puntos = [
            (cx + hw * cos - hh * sin, cy - hw * sin - hh * cos), (cx - hw * cos - hh * sin, cy + hw * sin - hh * cos),
            (cx - hw * cos + hh * sin, cy + hw * sin + hh * cos), (cx + hw * cos + hh * sin, cy - hw * sin + hh * cos)
        ]
        for px, py in puntos:
            if not (0 <= px < ANCHO_PANTALLA and 0 <= py < ALTO_PANTALLA) or pista_mask.get_at((int(px), int(py))) == COLOR_FONDO_CESPED:
                self.vivo = False
                print("¡Colisión!")
                break
    
    def actualizar_sensores(self, pista_mask):
        if not self.vivo: return
        self.sensores.clear()
        for ang_sensor in [-90, -45, 0, 45, 90]:
            longitud, ang_total = 0, self.angulo + ang_sensor
            x_ini, y_ini = int(self.posicion[0]), int(self.posicion[1])
            x_fin, y_fin = x_ini, y_ini
            while longitud < 350:
                longitud += 1
                x_fin = int(x_ini - math.sin(math.radians(ang_total)) * longitud)
                y_fin = int(y_ini - math.cos(math.radians(ang_total)) * longitud)
                if not (0 <= x_fin < ANCHO_PANTALLA and 0 <= y_fin < ALTO_PANTALLA) or pista_mask.get_at((x_fin, y_fin)) == COLOR_FONDO_CESPED:
                    break
            self.sensores.append([(x_ini, y_ini), (x_fin, y_fin)])
            
    def reiniciar(self):
        self.posicion = self.posicion_inicial[:]
        self.angulo = self.angulo_inicial
        self.vivo = True
        self.sprite = pygame.transform.rotozoom(self.sprite_original, self.angulo, 1)
        self.rect = self.sprite.get_rect(center=self.posicion)

# --- 3. BUCLE PRINCIPAL ---
mi_coche = Coche()
corriendo = True
while corriendo:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT: corriendo = False
        if evento.type == pygame.KEYDOWN and evento.key == pygame.K_SPACE:
            mi_coche.reiniciar()

    if mi_coche.vivo:
        teclas = pygame.key.get_pressed()
        if teclas[pygame.K_UP]: mi_coche.mover()
        if teclas[pygame.K_LEFT]: mi_coche.rotar(izquierda=True)
        if teclas[pygame.K_RIGHT]: mi_coche.rotar(derecha=True)

    mi_coche.actualizar_sensores(PISTA)
    mi_coche.comprobar_colision(PISTA)
    
    PANTALLA.blit(PISTA, (0, 0))
    mi_coche.dibujar(PANTALLA)
    
    pygame.display.flip()
    reloj.tick(FPS)

pygame.quit()
print("Simulación manual terminada.")