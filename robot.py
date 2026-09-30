import random
import time
import os


def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')


# ==========================================================
# ZONA DE TRABAJO (LA REGLA DE SUPERVIVENCIA)
# ==========================================================
def evaluar_robot(adn):
    posicion = [0, 0]  # Coordenadas iniciales: [fila, columna]
    piso_acido = False  # Bandera para registrar si pisa el ácido

    # El robot ejecuta su secuencia genética a ciegas
    for comando in adn:
        if comando == 'U':
            posicion[0] -= 1
        elif comando == 'D':
            posicion[0] += 1
        elif comando == 'L':
            posicion[1] -= 1
        elif comando == 'R':
            posicion[1] += 1

        # Verificación exacta de la coordenada [4, 4]
        if posicion == [4, 4]:
            piso_acido = True

    cubo = [7, 7]  # Coordenadas de la meta
    distancia = abs(cubo[0] - posicion[0]) + abs(cubo[1] - posicion[1])

    # El fitness base premia la proximidad a la meta
    puntaje = 100 - distancia

    # Penalización si tocó el ácido
    if piso_acido:
        puntaje -= 50

    return puntaje


# ==========================================================
# MOTOR GRAFICO (ACTUALIZADO CON DETECCIÓN EN VIVO)
# ==========================================================
def animar_mejor_robot(adn, generacion, puntaje_final):
    posicion, cubo, tamano = [0, 0], [7, 7], 8
    pos_acido = [4, 4]
    piso_acido = False

    for paso, comando in enumerate(adn):
        if comando == 'U':
            posicion[0] -= 1
        elif comando == 'D':
            posicion[0] += 1
        elif comando == 'L':
            posicion[1] -= 1
        elif comando == 'R':
            posicion[1] += 1

        # Verificamos si toca el ácido en la animación
        if posicion == pos_acido:
            piso_acido = True

        # Puntuación en tiempo real para la animación
        distancia_actual = abs(cubo[0] - posicion[0]) + abs(cubo[1] - posicion[1])
        puntaje_actual = 100 - distancia_actual
        if piso_acido:
            puntaje_actual -= 50

        # Limitar bordes para la representación gráfica
        pos_grafica_row = max(0, min(posicion[0], tamano - 1))
        pos_grafica_col = max(0, min(posicion[1], tamano - 1))

        limpiar_pantalla()
        print(f"Gen {generacion} | Puntos actual: {puntaje_actual}/100 {'⚠️ ¡ÁCIDO!' if piso_acido else ''}")

        for fila in range(tamano):
            linea = ""
            for columna in range(tamano):
                if fila == cubo[0] and columna == cubo[1]:
                    linea += "🎉" if [pos_grafica_row, pos_grafica_col] == cubo else "🟩"
                elif fila == pos_grafica_row and columna == pos_grafica_col:
                    linea += "🤖"
                elif fila == pos_acido[0] and columna == pos_acido[1]:
                    linea += "🟨"
                else:
                    linea += "⬜"
            print(linea)
        time.sleep(0.08)


# ==========================================================
# CICLO EVOLUTIVO
# ==========================================================
comandos = ['U', 'D', 'L', 'R']
mejor_robot = "".join(random.choice(comandos) for _ in range(30))
generacion = 0

while evaluar_robot(mejor_robot) < 100:
    robot_mutante = ""
    for gen in mejor_robot:
        robot_mutante += random.choice(comandos) if random.random() < 0.15 else gen

    if evaluar_robot(robot_mutante) > evaluar_robot(mejor_robot):
        mejor_robot = robot_mutante

    generacion += 1
    # Se anima el progreso cada 150 iteraciones
    if generacion % 150 == 0:
        animar_mejor_robot(mejor_robot, generacion, evaluar_robot(mejor_robot))

animar_mejor_robot(mejor_robot, generacion, evaluar_robot(mejor_robot))