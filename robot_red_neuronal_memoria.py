import argparse
import os
import random
import time
from copy import deepcopy

import torch
import torch.nn as nn


# ==========================================================
# 1. CONFIGURACION DEL PROBLEMA
# ==========================================================
# Este archivo deriva del laboratorio anterior y mantiene la estructura
# pedagogica del robot reactivo evolutivo. La diferencia principal es que
# la red ya no decide solo a partir de la observacion actual, sino a partir
# de una historia reciente: la observacion actual, la observacion anterior
# y la accion anterior. Esto permite que el robot tenga memoria de corto
# plazo y reduzca la probabilidad de quedar atrapado en un ciclo.
COMANDOS = ("U", "D", "L", "R")
ORDEN_SENSORES = ("U", "D", "L", "R")

TAMANO = 8
INICIO = (0, 0)
META = (7, 7)

OBSTACULOS = {
    (0, 3), (1, 3), (2, 0), (2, 2), (2, 3),
    (4, 5), (5, 0),(5, 2), (5, 3), (5, 4),
    (6,1), (6, 6), (5, 5), (4, 3),
}

PASOS_MAXIMOS = 30
TAMANO_POBLACION = 40
ELITE = 6
PADRES = 16

EMOJI_ROBOT = "\U0001f916"
EMOJI_META = "\U0001f7e9"
EMOJI_META_ALCANZADA = "\U0001f389"
EMOJI_OBSTACULO = "\u2b1b"
EMOJI_INICIO = "\U0001f535"
EMOJI_LIBRE = "\u2b1c"


# ==========================================================
# 2. MODELO NEURONAL CON MEMORIA SENSORIAL
# ==========================================================
# La entrada a la red ahora incluye tres componentes:
#   - sensado actual: 4 bits
#   - sensado anterior: 4 bits
#   - accion anterior codificada one-hot: 4 bits
#
# Resultado: 12 entradas totales porque cada una de estas tres partes aporta 4
# elementos. No es un numero arbitrario: cada bit representa una direccion
# cardinal (U, D, L, R), y la accion anterior tambien se codifica en 4 bits para
# que la red pueda distinguir entre mover arriba, abajo, izquierda o derecha.
#
# La idea pedagogica es introducir la dependencia temporal:
#    a_t = pi(s_t, s_{t-1}, a_{t-1})
#
# En otras palabras, el robot ya no decide solo con el presente, sino con la
# memoria de lo que vio y lo que hizo hace un instante. Esto es importante para
# detectar patrones repetitivos y reducir bucles sensoriomotores.
class RedPercepcionMemoria(nn.Module):
    """Red con memoria sensorial, 12 entradas y 4 acciones de salida."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden = nn.Linear(12, hidden_size)
        self.out = nn.Linear(hidden_size, 4)

    def forward(self, x):
        return self.out(torch.relu(self.hidden(x)))

    def vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def cargar_vector(self, vector):
        with torch.no_grad():
            pos = 0
            for parametro in self.parameters():
                largo = parametro.numel()
                parametro.copy_(vector[pos:pos + largo].reshape_as(parametro))
                pos += largo


def crear_red_aleatoria(rng, hidden_size=16):
    """Inicializa un individuo con pesos aleatorios."""
    red = RedPercepcionMemoria(hidden_size=hidden_size)
    for parametro in red.parameters():
        parametro.data.normal_(mean=0.0, std=0.6)
    return red


def copiar_red(red):
    """Crea una copia profunda de la red."""
    copia = RedPercepcionMemoria(hidden_size=16)
    copia.load_state_dict(deepcopy(red.state_dict()))
    return copia


def guardar_red(red, ruta):
    """Guarda la mejor red en disco para reutilizarla en otra prueba."""
    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    estado = {
        "state_dict": red.state_dict(),
        "hidden_size": 16,
        "comandos": COMANDOS,
        "orden_sensores": ORDEN_SENSORES,
        "tamano": TAMANO,
        "inicio": INICIO,
        "meta": META,
        "obstaculos": sorted(OBSTACULOS),
    }
    torch.save(estado, ruta)
    return ruta


def cargar_red(ruta):
    """Carga una red previamente guardada."""
    checkpoint = torch.load(ruta, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        return checkpoint

    red = RedPercepcionMemoria(hidden_size=checkpoint.get("hidden_size", 16))
    red.load_state_dict(checkpoint["state_dict"])
    red.eval()
    return red


# ==========================================================
# 3. ENTORNO Y SENSADO
# ==========================================================
def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


def destino_de(posicion, comando):
    """Devuelve la celda siguiente en la direccion indicada."""
    fila, columna = posicion
    cambios = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    cambio_fila, cambio_columna = cambios[comando]
    return fila + cambio_fila, columna + cambio_columna


def esta_bloqueada(posicion, obstaculos=None):
    """Verifica si una celda es invalida: fuera del tablero u obstaculo."""
    fila, columna = posicion
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    return not (0 <= fila < TAMANO and 0 <= columna < TAMANO) or posicion in obstaculos


def observar(posicion, obstaculos=None):
    """Codifica la observacion actual en 4 bits: U, D, L, R."""
    patron = 0
    for comando in ORDEN_SENSORES:
        patron <<= 1
        patron |= int(esta_bloqueada(destino_de(posicion, comando), obstaculos=obstaculos))
    return patron


def bits_de_observacion(observacion):
    """Convierte un entero de 4 bits en una lista [U, D, L, R]."""
    return [(observacion >> (3 - i)) & 1 for i in range(4)]


def accion_a_one_hot(comando):
    """Codifica la accion previa como one-hot en 4 valores."""
    vec = [0, 0, 0, 0]
    if comando in COMANDOS:
        vec[COMANDOS.index(comando)] = 1
    return vec


def vector_entrada(observacion_actual, observacion_anterior=None, accion_anterior=None):
    """
    Construye la entrada de la red con memoria sensorial.
    Entrada final = [obs_actual, obs_anterior, accion_anterior]
    = 4 + 4 + 4 = 12 elementos.
    """
    actual = bits_de_observacion(observacion_actual)
    anterior = bits_de_observacion(observacion_anterior) if observacion_anterior is not None else [0, 0, 0, 0]
    accion = accion_a_one_hot(accion_anterior) if accion_anterior is not None else [0, 0, 0, 0]
    return actual + anterior + accion


def decidir(red, posicion, obstaculos=None, observacion_anterior=None, accion_anterior=None):
    """Pasa la observacion actual y la memoria reciente a la red."""
    observacion_actual = observar(posicion, obstaculos=obstaculos)
    entrada = torch.tensor(
        vector_entrada(observacion_actual, observacion_anterior, accion_anterior),
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = red(entrada)[0]
    indice = int(torch.argmax(logits).item())
    return COMANDOS[indice], observacion_actual


def mover(posicion, comando, obstaculos=None, tamano=None):
    """Intenta mover al robot y devuelve si hubo avance o choque."""
    destino = destino_de(posicion, comando)
    tamano = TAMANO if tamano is None else tamano
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    if not (0 <= destino[0] < tamano and 0 <= destino[1] < tamano):
        return posicion, "borde"
    if destino in obstaculos:
        return posicion, "obstaculo"
    return destino, "avance"


def recorrer(red, obstaculos=None, inicio=None, meta=None, pasos_maximos=None, tamano=None):
    """Ejecuta una trayectoria completa para un individuo concreto."""
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    inicio = INICIO if inicio is None else inicio
    meta = META if meta is None else meta
    pasos_maximos = PASOS_MAXIMOS if pasos_maximos is None else pasos_maximos
    tamano = TAMANO if tamano is None else tamano

    posicion = inicio
    trayectoria = [posicion]
    observaciones = []
    acciones = []
    choques = 0
    pasos_utiles = 0

    observacion_anterior = None
    accion_anterior = None
    historial_posiciones = {posicion}
    ciclos = 0

    for _ in range(pasos_maximos):
        comando, observacion_actual = decidir(
            red,
            posicion,
            obstaculos=obstaculos,
            observacion_anterior=observacion_anterior,
            accion_anterior=accion_anterior,
        )
        nueva_posicion, resultado = mover(posicion, comando, obstaculos=obstaculos, tamano=tamano)

        observaciones.append(observacion_actual)
        acciones.append(comando)

        if resultado in ("borde", "obstaculo"):
            choques += 1
        if resultado == "avance":
            pasos_utiles += 1

        posicion = nueva_posicion
        trayectoria.append(posicion)

        # Penalizacion simple de bucle: si la red repite una posicion ya visitada,
        # se suma un contador. Esto refleja el problema del ciclo sensorimotor.
        if posicion in historial_posiciones:
            ciclos += 1
        historial_posiciones.add(posicion)

        if posicion == meta:
            break

        observacion_anterior = observacion_actual
        accion_anterior = comando

    return trayectoria, choques, pasos_utiles, observaciones, acciones, ciclos


# ==========================================================
# 4. VISUALIZACION DEL COMPORTAMIENTO
# ==========================================================
# Esta parte es importante pedagogicamente porque permite ver que la red no solo
# devuelve un valor numerico, sino que genera una secuencia de decisiones a lo
# largo del tiempo. Es la version didactica del comportamiento observable del
# robot en el mundo.
def dibujar(red, pausa=0.1, titulo="MEJOR INDIVIDUO", obstaculos=None, inicio=None, meta=None, pasos_maximos=None, tamano=None):
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    inicio = INICIO if inicio is None else inicio
    meta = META if meta is None else meta
    pasos_maximos = PASOS_MAXIMOS if pasos_maximos is None else pasos_maximos
    tamano = TAMANO if tamano is None else tamano

    trayectoria, _, _, observaciones, acciones, _ = recorrer(
        red,
        obstaculos=obstaculos,
        inicio=inicio,
        meta=meta,
        pasos_maximos=pasos_maximos,
        tamano=tamano,
    )

    if len(trayectoria) <= 1:
        return

    for paso, posicion in enumerate(trayectoria[1:], start=1):
        limpiar_pantalla()
        print(f"RED CON MEMORIA SENSORIAL | {titulo} | Fitness: {evaluar(red, obstaculos=obstaculos, inicio=inicio, meta=meta, pasos_maximos=pasos_maximos, tamano=tamano)}")
        print(f"{EMOJI_ROBOT} = robot | {EMOJI_META_ALCANZADA} = meta | {EMOJI_META} = objetivo | {EMOJI_OBSTACULO} = obstaculo")
        print(f"{EMOJI_INICIO} = inicio | {EMOJI_LIBRE} = libre\n")

        for fila in range(tamano):
            linea = ""
            for columna in range(tamano):
                celda = (fila, columna)
                if celda == posicion:
                    linea += EMOJI_ROBOT
                elif celda == inicio:
                    linea += EMOJI_INICIO
                elif celda == meta:
                    linea += EMOJI_META_ALCANZADA if posicion == meta else EMOJI_META
                elif celda in obstaculos:
                    linea += EMOJI_OBSTACULO
                else:
                    linea += EMOJI_LIBRE
            print(linea)

        print(f"Paso {paso}/{len(trayectoria)-1} | Observacion: {observaciones[paso - 1]:04b} | Accion: {acciones[paso - 1]} | Red con memoria")
        if pausa:
            time.sleep(pausa)


# ==========================================================
# 5. FITNESS Y EVOLUCION
# ==========================================================
def evaluar(red, obstaculos=None, inicio=None, meta=None, pasos_maximos=None, tamano=None):
    """Calcula el fitness del individuo."""
    obstaculos = OBSTACULOS if obstaculos is None else set(obstaculos)
    inicio = INICIO if inicio is None else inicio
    meta = META if meta is None else meta
    pasos_maximos = PASOS_MAXIMOS if pasos_maximos is None else pasos_maximos
    tamano = TAMANO if tamano is None else tamano

    trayectoria, choques, pasos_utiles, _, _, ciclos = recorrer(
        red,
        obstaculos=obstaculos,
        inicio=inicio,
        meta=meta,
        pasos_maximos=pasos_maximos,
        tamano=tamano,
    )

    if not trayectoria:
        return -1e9

    posicion_final = trayectoria[-1]
    distancia = abs(posicion_final[0] - meta[0]) + abs(posicion_final[1] - meta[1])
    llego = 1 if posicion_final == meta else 0

    # El fitness mantiene la idea del laboratorio anterior pero agrega una
    # penalizacion extra por repeticion de posiciones, lo cual ayuda a
    # favorecer politicas que no quedan atrapadas en bucles.
    fitness = 500 - 35 * distancia + 4 * pasos_utiles - 30 * choques + 2000 * llego - 10 * ciclos
    return fitness


def seleccionar_padres(poblacion, fitnesses, cantidad):
    """Selecciona una fraccion de los mejores individuos."""
    indices = list(range(len(poblacion)))
    orden = sorted(indices, key=lambda i: fitnesses[i], reverse=True)
    return [poblacion[i] for i in orden[:cantidad]]


def cruzar(red1, red2):
    """Cruce en un punto sobre el vector plano de pesos."""
    vector1 = red1.vector()
    vector2 = red2.vector()
    punto = random.randint(1, min(len(vector1), len(vector2)) - 1)

    hijo1 = torch.cat([vector1[:punto].clone(), vector2[punto:].clone()])
    hijo2 = torch.cat([vector2[:punto].clone(), vector1[punto:].clone()])

    nuevo1 = RedPercepcionMemoria(hidden_size=16)
    nuevo1.cargar_vector(hijo1)
    nuevo2 = RedPercepcionMemoria(hidden_size=16)
    nuevo2.cargar_vector(hijo2)
    return nuevo1, nuevo2


def mutar(red, std=0.16):
    """Mutacion gaussiana sobre pesos y sesgos."""
    with torch.no_grad():
        for parametro in red.parameters():
            parametro.add_(torch.randn_like(parametro) * std)
    return red


def evolucionar(generaciones=40, semilla=None, sin_pausa=False, guardar_mejor=None, cargar=None):
    """Ejecuta la evolucion del algoritmo genetico."""
    if semilla is not None:
        random.seed(semilla)
        torch.manual_seed(semilla)

    if cargar is not None:
        mejor = cargar_red(cargar)
        return mejor

    poblacion = [crear_red_aleatoria(random) for _ in range(TAMANO_POBLACION)]
    mejor = None

    for _ in range(generaciones):
        fitnesses = [evaluar(red) for red in poblacion]
        orden = sorted(range(len(poblacion)), key=lambda i: fitnesses[i], reverse=True)
        mejor_actual = poblacion[orden[0]]

        if mejor is None or fitnesses[orden[0]] > evaluar(mejor):
            mejor = copiar_red(mejor_actual)

        elites = [copiar_red(poblacion[i]) for i in orden[:ELITE]]
        padres = seleccionar_padres(poblacion, fitnesses, PADRES)

        nueva_poblacion = elites[:]
        while len(nueva_poblacion) < TAMANO_POBLACION:
            padre_a = random.choice(padres)
            padre_b = random.choice(padres)
            hijo1, hijo2 = cruzar(padre_a, padre_b)
            mutar(hijo1, std=0.20)
            mutar(hijo2, std=0.20)
            nueva_poblacion.extend([hijo1, hijo2])

        poblacion = nueva_poblacion[:TAMANO_POBLACION]

        if not sin_pausa:
            time.sleep(0.05)

    if guardar_mejor:
        guardar_red(mejor, guardar_mejor)

    return mejor


def parse_args():
    parser = argparse.ArgumentParser(description="Laboratorio IA 8: red neuronal con memoria sensorial")
    parser.add_argument("--generaciones", type=int, default=40)
    parser.add_argument("--semilla", type=int, default=None)
    parser.add_argument("--sin-pausa", action="store_true")
    parser.add_argument("--guardar-mejor", type=str, default=None)
    parser.add_argument("--cargar-red", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mejor = evolucionar(
        generaciones=args.generaciones,
        semilla=args.semilla,
        sin_pausa=args.sin_pausa,
        guardar_mejor=args.guardar_mejor,
        cargar=args.cargar_red,
    )

    if mejor is not None:
        trayectoria, choques, pasos_utiles, observaciones, acciones, ciclos = recorrer(mejor)
        llego = trayectoria[-1] == META
        
        print("\n--- MÉTRICAS DETALLADAS ---")
        print(f"¿Llegó a la meta?: {'Sí' if llego else 'No'}")
        print(f"Pasos usados: {len(trayectoria) - 1}")
        print(f"Pasos útiles: {pasos_utiles}")
        print(f"Choques producidos: {choques}")
        print(f"Ciclos / bucles detectados: {ciclos}")
        print(f"Fitness final: {evaluar(mejor)}")

    if not args.sin_pausa:
        print("\nANIMACION DEL MEJOR INDIVIDUO")
        dibujar(mejor, pausa=0.08, titulo="MEJOR INDIVIDUO")