import argparse
import os
import random
import time

COMANDOS = ("U", "D", "L", "R")
TAMANO = 8
META = (7, 7)
OBSTACULO = (4, 4)  # Reto 5: Definición del obstáculo
LONGITUD_ADN = 14


def limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


def recorrer(adn):
    fila, columna = 0, 0
    trayectoria = [(fila, columna)]

    for comando in adn:
        if comando == "U":
            fila = max(0, fila - 1)
        elif comando == "D":
            fila = min(TAMANO - 1, fila + 1)
        elif comando == "L":
            columna = max(0, columna - 1)
        elif comando == "R":
            columna = min(TAMANO - 1, columna + 1)
        trayectoria.append((fila, columna))

    return trayectoria


def evaluar(adn):
    trayectoria = recorrer(adn)
    posicion = trayectoria[-1]
    distancia = abs(META[0] - posicion[0]) + abs(META[1] - posicion[1])
    pasos_utiles = sum(
        actual != anterior
        for anterior, actual in zip(trayectoria, trayectoria[1:])
    )
    visitas_repetidas = len(trayectoria) - len(set(trayectoria))

    puntaje = 1000 - distancia * 40
    puntaje += pasos_utiles * 2
    puntaje -= visitas_repetidas

    # Reto 5: Penalización si la trayectoria pisa el obstáculo en (4,4)
    if OBSTACULO in trayectoria:
        puntaje -= 500

    if posicion == META:
        puntaje += 1000
    return puntaje


def diversidad(poblacion):
    return len({"".join(adn) for adn in poblacion})


def mutar(adn, tasa_mutacion, rng):
    return [
        rng.choice(COMANDOS) if rng.random() < tasa_mutacion else comando
        for comando in adn
    ]


def evolucionar(nombre, tasa_mutacion, semilla, maximo_generaciones=2500):
    rng = random.Random(semilla)
    poblacion = [
        [rng.choice(COMANDOS) for _ in range(LONGITUD_ADN)]
        for _ in range(80)
    ]
    historial = []

    for generacion in range(maximo_generaciones + 1):
        poblacion.sort(key=evaluar, reverse=True)
        mejor = poblacion[0]
        puntaje = evaluar(mejor)
        historial.append((generacion, puntaje, diversidad(poblacion)))

        if recorrer(mejor)[-1] == META:
            return {
                "nombre": nombre,
                "adn": mejor,
                "generacion": generacion,
                "puntaje": puntaje,
                "diversidad": diversidad(poblacion),
                "historial": historial,
            }

        elite = poblacion[:8]
        nueva_poblacion = [robot[:] for robot in elite]
        while len(nueva_poblacion) < len(poblacion):
            padre = rng.choice(poblacion[:30])
            nueva_poblacion.append(mutar(padre, tasa_mutacion, rng))
        poblacion = nueva_poblacion

    return {
        "nombre": nombre,
        "adn": poblacion[0],
        "generacion": maximo_generaciones,
        "puntaje": evaluar(poblacion[0]),
        "diversidad": diversidad(poblacion),
        "historial": historial,
    }


def dibujar(adn, titulo, generacion, puntaje, pausa):
    trayectoria = recorrer(adn)
    for paso, posicion in enumerate(trayectoria[1:], start=1):
        limpiar_pantalla()
        print(f"{titulo} | Gen {generacion} | Puntos: {puntaje}")
        print("Exploracion: muta; explotacion: conserva.\n")
        for fila in range(TAMANO):
            linea = ""
            for columna in range(TAMANO):
                celda = (fila, columna)
                if celda == META:
                    linea += "🎉" if posicion == META else "🟩"
                elif celda == posicion:
                    linea += "🤖"
                elif celda == OBSTACULO:
                    linea += "🟥"  # Representación visual del obstáculo
                else:
                    linea += "⬜"
            print(linea)
        print(f"Paso {paso}/{len(adn)} | ADN: {''.join(adn)}")
        if pausa:
            time.sleep(pausa)


def mostrar_resultado(resultado, pausa):
    dibujar(
        resultado["adn"],
        resultado["nombre"],
        resultado["generacion"],
        resultado["puntaje"],
        0 if pausa == 0 else 0.08,
    )
    print(f"Diversidad final: {resultado['diversidad']}/80")
    if pausa:
        time.sleep(1.5)


def main():
    parser = argparse.ArgumentParser(
        description="Compara exploracion y explotacion en una poblacion de robots."
    )
    parser.add_argument(
        "--modo",
        choices=("comparar", "exploracion", "explotacion"),
        default="comparar",
    )
    parser.add_argument("--semilla", type=int, default=7)
    parser.add_argument("--sin-pausa", action="store_true")
    args = parser.parse_args()
    pausa = 0 if args.sin_pausa else 1.5

    experimentos = {
        "exploracion": ("EXPLORACION", 0.30),
        "explotacion": ("EXPLOTACION", 0.03),
    }
    modos = (args.modo,) if args.modo != "comparar" else experimentos
    resultados = []

    for indice, modo in enumerate(modos):
        nombre, tasa = experimentos[modo]
        resultado = evolucionar(nombre, tasa, args.semilla + indice)
        resultados.append(resultado)
        mostrar_resultado(resultado, pausa)

    if len(resultados) == 2:
        print("\nCOMPARACION")
        print("Modo          Mutacion  Generaciones  Diversidad final")
        print("-------------------------------------------------------")
        for resultado in resultados:
            modo = resultado["nombre"].lower()
            tasa = experimentos[modo][1]
            print(
                f"{modo:<14}{tasa:<10.2f}{resultado['generacion']:<14}"
                f"{resultado['diversidad']}/80"
            )


if __name__ == "__main__":
    main()