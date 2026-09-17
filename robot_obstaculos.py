import argparse
import os
import random
import time

# ==========================================================
# 1. CONFIGURACION DEL PROBLEMA
# ==========================================================
# Cada robot tiene un ADN formado por comandos de movimiento.
COMANDOS = ("U", "D", "L", "R")

# La cuadricula usa coordenadas (fila, columna), empezando en (0, 0).
TAMANO = 8
INICIO = (0, 0)
META = (7, 7)

# Estas celdas estan bloqueadas. El robot no puede entrar en ellas.
OBSTACULOS = {
    (0, 3), (1, 3), (2, 1), (2, 2), (2, 3),
    (3, 5), (4, 5), (5, 2), (5, 3), (5, 4),
    (6, 6),
}

# Parametros de la poblacion y de los individuos.
LONGITUD_ADN = 18
TAMANO_POBLACION = 100
ELITE = 10
PADRES = 40

# ----------------------------------------------------------
# RETO DE MODIFICACION (punto 6 del manual)
# Se cambia UNA sola penalizacion del fitness y se conservan
# las demas condiciones del experimento.
#
# Penalizacion original por choque: -30
# Penalizacion nueva por choque:    -80
#
# Justificacion: se busca una presion selectiva mucho mas
# fuerte contra los choques, para comprobar (pregunta 3 del
# manual) si una penalizacion grande reduce los choques a
# cambio de mas generaciones para converger, o si vuelve al
# algoritmo demasiado conservador cerca de los obstaculos.
# ----------------------------------------------------------
PENALIZACION_CHOQUE = 80  # antes: 30

# Los escapes Unicode mantienen el archivo en ASCII, pero la terminal
# sigue mostrando los mismos emojis que en las clases anteriores.
EMOJI_ROBOT = "\U0001f916"
EMOJI_META = "\U0001f7e9"
EMOJI_META_ALCANZADA = "\U0001f389"
EMOJI_OBSTACULO = "\u2b1b"
EMOJI_INICIO = "\U0001f535"
EMOJI_LIBRE = "\u2b1c"


# ==========================================================
# 2. SIMULACION DEL ENTORNO
# ==========================================================
def limpiar_pantalla():
    """Limpia la terminal antes de dibujar el siguiente paso."""
    os.system("cls" if os.name == "nt" else "clear")


def mover(posicion, comando):
    """Intenta ejecutar un comando y devuelve posicion y resultado."""
    fila, columna = posicion
    cambios = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    cambio_fila, cambio_columna = cambios[comando]
    destino = (fila + cambio_fila, columna + cambio_columna)

    # Un movimiento fuera del tablero no cambia la posicion.
    if not (0 <= destino[0] < TAMANO and 0 <= destino[1] < TAMANO):
        return posicion, "borde"

    # Chocar con un obstaculo tampoco permite avanzar.
    if destino in OBSTACULOS:
        return posicion, "obstaculo"

    return destino, "avance"


def recorrer(adn):
    """Ejecuta un ADN y registra trayectoria, choques y pasos utiles."""
    posicion = INICIO
    trayectoria = [posicion]
    choques = 0
    pasos_utiles = 0

    for comando in adn:
        nueva_posicion, resultado = mover(posicion, comando)
        if resultado == "obstaculo":
            choques += 1
        if resultado == "avance":
            pasos_utiles += 1
        posicion = nueva_posicion
        trayectoria.append(posicion)

    return trayectoria, choques, pasos_utiles


# ==========================================================
# 3. REGLA DE SUPERVIVENCIA: EL FITNESS
# ==========================================================
def evaluar(adn):
    """Calcula el fitness de un robot: cuanto mayor, mejor solucion."""
    trayectoria, choques, pasos_utiles = recorrer(adn)
    posicion = trayectoria[-1]
    distancia = abs(META[0] - posicion[0]) + abs(META[1] - posicion[1])
    visitas_repetidas = len(trayectoria) - len(set(trayectoria))

    # Los pesos son decisiones del disenador y pueden modificarse.
    puntaje = 500
    puntaje -= distancia * 35
    puntaje += pasos_utiles * 4
    puntaje -= choques * PENALIZACION_CHOQUE  # <-- linea modificada (antes: choques * 30)
    puntaje -= visitas_repetidas * 3

    # Llegar vale mucho mas que estar simplemente cerca.
    if posicion == META:
        puntaje += 2000

    return puntaje


def diversidad(poblacion):
    """Cuenta cuantos ADN diferentes hay en la poblacion."""
    return len({"".join(adn) for adn in poblacion})


# ==========================================================
# 4. OPERADORES GENETICOS
# ==========================================================
def mutar(adn, tasa_mutacion, rng):
    """Cambia algunos genes al azar segun la tasa de mutacion."""
    return [
        rng.choice(COMANDOS) if rng.random() < tasa_mutacion else comando
        for comando in adn
    ]


def cruzar(primer_padre, segundo_padre, rng):
    """Combina el comienzo de un padre y el final del otro."""
    punto = rng.randint(1, len(primer_padre) - 1)
    return primer_padre[:punto] + segundo_padre[punto:]


# ==========================================================
# 5. CICLO EVOLUTIVO
# ==========================================================
def evolucionar(tasa_mutacion, usar_cruce, semilla, maximo_generaciones=3000):
    """Ejecuta el algoritmo genetico y devuelve el mejor resultado."""
    rng = random.Random(semilla)

    # La primera generacion es completamente aleatoria.
    poblacion = [
        [rng.choice(COMANDOS) for _ in range(LONGITUD_ADN)]
        for _ in range(TAMANO_POBLACION)
    ]

    historial = []

    for generacion in range(maximo_generaciones + 1):
        # El primer robot sera el mejor despues de ordenar la poblacion.
        poblacion.sort(key=evaluar, reverse=True)
        mejor = poblacion[0]
        puntaje = evaluar(mejor)
        trayectoria, choques, pasos_utiles = recorrer(mejor)
        historial.append((generacion, puntaje, diversidad(poblacion)))

        # Terminamos temprano si ya encontramos una ruta valida.
        if trayectoria[-1] == META:
            return resultado(
                usar_cruce, mejor, generacion, puntaje, diversidad(poblacion),
                choques, pasos_utiles, historial,
            )

        # El elitismo copia directamente a los mejores robots.
        nueva_poblacion = [robot[:] for robot in poblacion[:ELITE]]

        while len(nueva_poblacion) < TAMANO_POBLACION:
            # Elegimos padres de la parte superior de la poblacion.
            primer_padre = rng.choice(poblacion[:PADRES])
            if usar_cruce:
                segundo_padre = rng.choice(poblacion[:PADRES])
                hijo = cruzar(primer_padre, segundo_padre, rng)
            else:
                hijo = primer_padre[:]

            # El cruce combina; la mutacion introduce variacion nueva.
            nueva_poblacion.append(mutar(hijo, tasa_mutacion, rng))

        poblacion = nueva_poblacion

    # Si se agotan las generaciones, devolvemos el mejor disponible.
    poblacion.sort(key=evaluar, reverse=True)
    mejor = poblacion[0]
    trayectoria, choques, pasos_utiles = recorrer(mejor)
    return resultado(
        usar_cruce, mejor, maximo_generaciones, evaluar(mejor),
        diversidad(poblacion), choques, pasos_utiles, historial,
    )


def resultado(usar_cruce, adn, generacion, puntaje, diversidad_final,
              choques, pasos_utiles, historial):
    """Reune las metricas que compararemos en el experimento."""
    trayectoria, _, _ = recorrer(adn)
    return {
        "metodo": "mutacion + cruce" if usar_cruce else "solo mutacion",
        "adn": adn,
        "generacion": generacion,
        "puntaje": puntaje,
        "diversidad": diversidad_final,
        "choques": choques,
        "pasos_utiles": pasos_utiles,
        "llego": trayectoria[-1] == META,
        "historial": historial,
    }


# ==========================================================
# 6. VISUALIZACION Y CLI
# ==========================================================
def dibujar(resultado, pausa):
    """Anima la trayectoria del mejor robot sobre la cuadricula."""
    adn = resultado["adn"]
    trayectoria, _, _ = recorrer(adn)

    for paso, posicion in enumerate(trayectoria[1:], start=1):
        limpiar_pantalla()
        print(
            f"{resultado['metodo'].upper()} | "
            f"Gen {resultado['generacion']} | Puntos: {resultado['puntaje']}"
        )
        print(
            f"{EMOJI_ROBOT} = robot | {EMOJI_META_ALCANZADA} = meta alcanzada | "
            f"{EMOJI_META} = meta | {EMOJI_OBSTACULO} = obstaculo"
        )
        print(f"{EMOJI_INICIO} = inicio | {EMOJI_LIBRE} = celda libre\n")

        for fila in range(TAMANO):
            linea = ""
            for columna in range(TAMANO):
                celda = (fila, columna)
                if celda == posicion:
                    linea += EMOJI_ROBOT
                elif celda == INICIO:
                    linea += EMOJI_INICIO
                elif celda == META:
                    linea += EMOJI_META_ALCANZADA if posicion == META else EMOJI_META
                elif celda in OBSTACULOS:
                    linea += EMOJI_OBSTACULO
                else:
                    linea += EMOJI_LIBRE
            print(linea)

        print(
            f"Paso {paso}/{len(adn)} | ADN: {''.join(adn)} | "
            f"Choques: {resultado['choques']}"
        )

        if pausa:
            time.sleep(pausa)


def mostrar_resultado(resultado, pausa):
    """Muestra la animacion y las metricas finales de una variante."""
    dibujar(resultado, pausa)
    estado = "LLEGO" if resultado["llego"] else "NO LLEGO"
    print(f"\nResultado: {estado}")
    print(f"Pasos utiles: {resultado['pasos_utiles']}")
    print(f"Diversidad final: {resultado['diversidad']}/{TAMANO_POBLACION}")
    if pausa:
        time.sleep(1)


def main():
    """Lee las opciones de terminal y ejecuta el experimento solicitado."""
    parser = argparse.ArgumentParser(
        description="Compara mutacion y cruce en una poblacion de robots."
    )
    parser.add_argument(
        "--modo",
        choices=("comparar", "mutacion", "cruce"),
        default="comparar",
    )
    parser.add_argument("--semilla", type=int, default=7)
    parser.add_argument("--sin-pausa", action="store_true")
    args = parser.parse_args()

    # comparar ejecuta ambas variantes; los otros modos ejecutan solo una.
    experimentos = {
        "mutacion": ("solo mutacion", False),
        "cruce": ("mutacion + cruce", True),
    }
    modos = tuple(experimentos) if args.modo == "comparar" else (args.modo,)

    resultados = []
    for indice, modo in enumerate(modos):
        _, usar_cruce = experimentos[modo]
        # Al comparar, usamos semillas distintas para las dos variantes.
        resultado_actual = evolucionar(
            tasa_mutacion=0.08,
            usar_cruce=usar_cruce,
            semilla=args.semilla + indice,
        )
        resultados.append(resultado_actual)
        mostrar_resultado(resultado_actual, 0 if args.sin_pausa else 0.08)

    if len(resultados) == 2:
        print("\nCOMPARACION")
        print("Metodo               Llego  Generaciones   Puntaje  Choques  Diversidad")
        print("-----------------------------------------------------------------------")
        for resultado_actual in resultados:
            llego = "si" if resultado_actual["llego"] else "no"
            print(
                f"{resultado_actual['metodo']:<20}{llego:<7}"
                f"{resultado_actual['generacion']:<15}"
                f"{resultado_actual['puntaje']:<9}"
                f"{resultado_actual['choques']:<9}"
                f"{resultado_actual['diversidad']}/{TAMANO_POBLACION}"
            )


if __name__ == "__main__":
    main()