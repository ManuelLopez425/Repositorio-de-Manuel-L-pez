import argparse
import json
import os
import random
import re
import time


# ==========================================================
# 1. CONFIGURACION DEL PROBLEMA
# ==========================================================
COMANDOS = ("U", "D", "L", "R")
ORDEN_SENSORES = ("U", "D", "L", "R")

TAMANO = 8
INICIO = (0, 0)
META = (7, 7)

OBSTACULOS = {
    (0, 3), (1, 3), (2, 0), (2, 2), (2, 3),
    (4, 5), (5, 2), (5, 3), (5, 4),
    (6, 6),
}

# ==========================================================
# [MODIFICACIÓN 4]: Segunda distribución de obstáculos opcional
# Para cumplir con el punto 4 del reto, definimos un segundo mapa 
# con una distribución alternativa de paredes/obstáculos para comparar el desempeño.
# ==========================================================
OBSTACULOS_MAPA_2 = {
    (1, 1), (1, 2), (3, 3), (3, 4), (3, 5),
    (5, 1), (5, 6), (6, 2), (6, 3), (4, 1),
}
# Variable global para alternar mapas según se requiera en la simulación
USAR_SEGUNDO_MAPA = False


# ==========================================================
# [MODIFICACIÓN 1]: Dirección relativa de la meta en la percepción
# Originalmente el ADN tenía 16 reglas (2^4 por los 4 sensores locales: U, D, L, R).
# Ahora añadimos 2 bits adicionales que indican si la meta está Generalmente Arriba/Abajo 
# y Generalmente Izquierda/Derecha respecto a la posición actual del robot.
# Esto expande las entradas de percepción de 4 bits a 6 bits -> 2^6 = 64 reglas en el ADN.
# ==========================================================
# [MODIFICACIÓN 2]: Memoria de la última acción
# Añadimos además 2 bits para codificar cuál fue la última acción realizada ("U", "D", "L", "R" o ninguna),
# evitando así ciclos infinitos inmediatos de estancamiento.
# Total de bits de percepción combinados = 4 (sensores) + 2 (dirección meta) + 2 (memoria última acción) = 8 bits.
# Longitud del ADN resultante = 2^8 = 256 reglas.
# ==========================================================
LONGITUD_ADN = 256  # Actualizado de 16 a 256 por el incremento de entradas sensoriales y memoria

# ==========================================================
# [MODIFICACIÓN 3]: Cambio del número máximo de pasos
# Modificamos los pasos máximos permitidos por ejecución. Un límite muy bajo 
# impide que el robot llegue si toma desvíos; un límite muy alto fomenta ciclos infinitos 
# si la política no converge correctamente a la meta. Lo ampliamos a 50 para darle oportunidad 
# al agente ampliado con memoria de maniobrar complejos laberintos.
# ==========================================================
PASOS_MAXIMOS = 50

TAMANO_POBLACION = 100
ELITE = 10
PADRES = 40

EMOJI_ROBOT = "\U0001f916"
EMOJI_META = "\U0001f7e9"
EMOJI_META_ALCANZADA = "\U0001f389"
EMOJI_OBSTACULO = "\u2b1b"
EMOJI_INICIO = "\U0001f535"
EMOJI_LIBRE = "\u2b1c"


# ==========================================================
# 2. PERCEPCION Y SIMULACION DEL ENTORNO
# ==========================================================
def limpiar_pantalla():
    """Limpia la terminal antes de dibujar el siguiente paso."""
    os.system("cls" if os.name == "nt" else "clear")


def destino_de(posicion, comando):
    """Devuelve la celda que queda en la direccion indicada."""
    fila, columna = posicion
    cambios = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    cambio_fila, cambio_columna = cambios[comando]
    return fila + cambio_fila, columna + cambio_columna


def esta_bloqueada(posicion):
    """Indica si una celda no puede ser ocupada por el robot (considera el mapa activo)."""
    fila, columna = posicion
    obstaculos_activos = OBSTACULOS_MAPA_2 if USAR_SEGUNDO_MAPA else OBSTACULOS
    return not (0 <= fila < TAMANO and 0 <= columna < TAMANO) or posicion in obstaculos_activos


def observar(posicion, ultima_accion):
    """
    [MODIFICACIÓN 1 y 2 COMBINADAS EN EL SISTEMA DE PERCEPCIÓN EXTENDIDA]
    Codifica un patrón binario extendido de 8 bits:
    - Bits 0-3: Sensores locales tradicionales (U, D, L, R).
    - Bits 4-5: Dirección relativa de la meta (Vertical: Arriba/Abajo, Horizontal: Izq/Der).
    - Bits 6-7: Codificación de la última acción ejecutada para evitar retrocesos en bucle.
    """
    # 1. Sensores locales (4 bits)
    patron_sensores = 0
    for comando in ORDEN_SENSORES:
        patron_sensores <<= 1
        patron_sensores |= int(esta_bloqueada(destino_de(posicion, comando)))

    # 2. Dirección relativa de la meta (2 bits)
    # Bit más significativo vertical (1 si la meta está abajo, 0 si está arriba/igual)
    dir_v = 1 if META[0] > posicion[0] else (2 if META[0] < posicion[0] else 0)
    # Bit horizontal (1 si la meta está a la derecha, 0 si está a la izquierda/igual)
    dir_h = 1 if META[1] > posicion[1] else (2 if META[1] < posicion[1] else 0)
    
    # Comprimir dirección de meta en 2 bits simplificados (0 a 3)
    meta_codificada = 0
    if META[0] > posicion[0]: meta_codificada |= 2
    if META[1] > posicion[1]: meta_codificada |= 1

    # 3. Memoria de la última acción (2 bits)
    # Mapeo de comandos previos: None=0, 'U'=1, 'D'=2, 'L'=3, 'R'=4 -> truncado a 2 bits (0-3)
    mapeo_accion = {None: 0, "U": 1, "D": 2, "L": 3, "R": 0}
    accion_codificada = mapeo_accion.get(ultima_accion, 0)

    # Concatenación total en un solo entero de 8 bits (0 a 255)
    patron_total = (patron_sensores << 4) | (meta_codificada << 2) | accion_codificada
    return patron_total


def decidir(adn, posicion, ultima_accion):
    """Consulta la regla del ADN utilizando la percepción extendida de 8 bits."""
    indice_observacion = observar(posicion, ultima_accion)
    return adn[indice_observacion]


def mover(posicion, comando):
    """Intenta ejecutar un comando y devuelve la nueva posición y el resultado."""
    destino = destino_de(posicion, comando)
    obstaculos_activos = OBSTACULOS_MAPA_2 if USAR_SEGUNDO_MAPA else OBSTACULOS
    if not (0 <= destino[0] < TAMANO and 0 <= destino[1] < TAMANO):
        return posicion, "borde"
    if destino in obstaculos_activos:
        return posicion, "obstaculo"
    return destino, "avance"


def recorrer(adn):
    """Ejecuta una política reactiva mejorada con memoria y supervisa su trayectoria."""
    posicion = INICIO
    trayectoria = [posicion]
    observaciones = []
    acciones = []
    choques = 0
    pasos_utiles = 0
    ultima_accion = None

    for _ in range(PASOS_MAXIMOS):
        observacion = observar(posicion, ultima_accion)
        comando = decidir(adn, posicion, ultima_accion)
        nueva_posicion, resultado = mover(posicion, comando)
        
        observaciones.append(observacion)
        acciones.append(comando)
        
        if resultado in ("borde", "obstaculo"):
            choques += 1
        if resultado == "avance":
            pasos_utiles += 1
            
        posicion = nueva_posicion
        trayectoria.append(posicion)
        ultima_accion = comando  # Actualiza la memoria para el siguiente ciclo
        
        if posicion == META:
            break

    return trayectoria, choques, pasos_utiles, observaciones, acciones


# ==========================================================
# 3. REGLA DE SUPERVIVENCIA: EL FITNESS
# ==========================================================
def evaluar(adn):
    """Calcula el fitness de la política con percepciones y memoria ampliadas."""
    trayectoria, choques, pasos_utiles, _, _ = recorrer(adn)
    posicion = trayectoria[-1]
    distancia = abs(META[0] - posicion[0]) + abs(META[1] - posicion[1])
    visitas_repetidas = len(trayectoria) - len(set(trayectoria))

    puntaje = 800
    puntaje -= distancia * 25
    puntaje += pasos_utiles * 6
    puntaje -= choques * 20
    puntaje -= visitas_repetidas * 4

    if posicion == META:
        puntaje += 3000
    return puntaje


def diversidad(poblacion):
    """Cuenta cuántas políticas diferentes existen en la población."""
    return len({"".join(adn) for adn in poblacion})


def reglas_de_politica(adn):
    """Devuelve la política con sus observaciones en formato binario de 8 bits como claves."""
    return {
        f"{indice:08b}": comando
        for indice, comando in enumerate(adn)
    }


# ==========================================================
# 4. OPERADORES GENETICOS
# ==========================================================
def mutar(adn, tasa_mutacion, rng):
    """Modifica aleatoriamente genes de la política según la tasa de mutación."""
    return [
        rng.choice(COMANDOS) if rng.random() < tasa_mutacion else comando
        for comando in adn
    ]


def cruzar(primer_padre, segundo_padre, rng):
    """Cruza de un punto partes de las tablas de reglas de los padres."""
    punto = rng.randint(1, len(primer_padre) - 1)
    return primer_padre[:punto] + segundo_padre[punto:]


# ==========================================================
# 5. CICLO EVOLUTIVO
# ==========================================================
def evolucionar(tasa_mutacion, usar_cruce, semilla, maximo_generaciones=3000):
    """Ejecuta el ciclo evolutivo del algoritmo genético con ADN extendido a 256."""
    rng = random.Random(semilla)
    poblacion = [
        [rng.choice(COMANDOS) for _ in range(LONGITUD_ADN)]
        for _ in range(TAMANO_POBLACION)
    ]
    historial = []

    for generacion in range(maximo_generaciones + 1):
        poblacion.sort(key=evaluar, reverse=True)
        mejor = poblacion[0]
        puntaje = evaluar(mejor)
        trayectoria, choques, pasos_utiles, _, _ = recorrer(mejor)
        historial.append((generacion, puntaje, diversidad(poblacion)))

        if trayectoria[-1] == META:
            return resultado(
                usar_cruce, mejor, generacion, puntaje, diversidad(poblacion),
                choques, pasos_utiles, historial, poblacion,
            )

        nueva_poblacion = [robot[:] for robot in poblacion[:ELITE]]
        while len(nueva_poblacion) < TAMANO_POBLACION:
            primer_padre = rng.choice(poblacion[:PADRES])
            if usar_cruce:
                segundo_padre = rng.choice(poblacion[:PADRES])
                hijo = cruzar(primer_padre, segundo_padre, rng)
            else:
                hijo = primer_padre[:]
            nueva_poblacion.append(mutar(hijo, tasa_mutacion, rng))
        poblacion = nueva_poblacion

    poblacion.sort(key=evaluar, reverse=True)
    mejor = poblacion[0]
    trayectoria, choques, pasos_utiles, _, _ = recorrer(mejor)
    return resultado(
        usar_cruce, mejor, maximo_generaciones, evaluar(mejor),
        diversidad(poblacion), choques, pasos_utiles, historial, poblacion,
    )


def resultado(usar_cruce, adn, generacion, puntaje, diversidad_final,
              choques, pasos_utiles, historial, poblacion):
    """Reúne las métricas del experimento."""
    trayectoria, _, _, _, _ = recorrer(adn)
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
        "poblacion": [robot[:] for robot in poblacion],
    }


def guardar_generacion(resultado_actual, identificador, semilla):
    """Guarda la última población en un archivo JSON estructurado."""
    identificador_limpio = re.sub(r"[^A-Za-z0-9_.-]+", "_", identificador)
    nombre = f"generacion{resultado_actual['generacion']}_{identificador_limpio}.txt"
    datos = {
        "version": 3,
        "generacion": resultado_actual["generacion"],
        "identificador": identificador_limpio,
        "semilla": semilla,
        "metodo": resultado_actual["metodo"],
        "longitud_adn": LONGITUD_ADN,
        "orden_sensores": "U D L R + Meta + Memoria",
        "poblacion": [
            {
                "indice": indice,
                "adn": "".join(robot),
                "reglas": reglas_de_politica(robot),
            }
            for indice, robot in enumerate(resultado_actual["poblacion"])
        ],
    }
    with open(nombre, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, indent=2)
    return nombre


def cargar_generacion(ruta):
    """Carga y valida una población guardada previamente."""
    with open(ruta, encoding="utf-8") as archivo:
        datos = json.load(archivo)

    poblacion = datos.get("poblacion")
    if not isinstance(poblacion, list) or not poblacion:
        raise ValueError("El archivo no contiene una población válida.")
    if datos.get("longitud_adn") != LONGITUD_ADN:
        raise ValueError("La longitud del ADN no coincide con esta versión modificada.")
    politicas = [
        individuo["adn"] if isinstance(individuo, dict) else individuo
        for individuo in poblacion
    ]
    return {
        "generacion": datos.get("generacion", "desconocida"),
        "identificador": datos.get("identificador", os.path.basename(ruta)),
        "metodo": datos.get("metodo", "generacion cargada"),
        "poblacion": [list(adn) for adn in politicas],
    }


def resultado_de_politica(adn, metodo, generacion, poblacion):
    """Construye las métricas para animar una política cargada."""
    trayectoria, choques, pasos_utiles, _, _ = recorrer(adn)
    return {
        "metodo": metodo,
        "adn": adn,
        "generacion": generacion,
        "puntaje": evaluar(adn),
        "diversidad": diversidad(poblacion),
        "choques": choques,
        "pasos_utiles": pasos_utiles,
        "llego": trayectoria[-1] == META,
        "historial": [],
        "poblacion": [robot[:] for robot in poblacion],
    }


def ejecutar_poblacion(datos, modo_ejecucion):
    """Selecciona el mejor individuo o devuelve todos para animarlos."""
    poblacion = datos["poblacion"]
    poblacion_ordenada = sorted(poblacion, key=evaluar, reverse=True)
    if modo_ejecucion == "mejor":
        return [resultado_de_politica(
            poblacion_ordenada[0], datos["metodo"], datos["generacion"], poblacion
        )]
    return [resultado_de_politica(
        adn, datos["metodo"], datos["generacion"], poblacion
    ) for adn in poblacion]


# ==========================================================
# 6. VISUALIZACION Y CLI
# ==========================================================
def dibujar(resultado, pausa):
    """Anima la trayectoria del robot en el mapa activo seleccionado."""
    adn = resultado["adn"]
    trayectoria, _, _, observaciones, acciones = recorrer(adn)
    obstaculos_activos = OBSTACULOS_MAPA_2 if USAR_SEGUNDO_MAPA else OBSTACULOS

    for paso, posicion in enumerate(trayectoria[1:], start=1):
        limpiar_pantalla()
        mapa_etiqueta = "MAPA 2 (Modificado)" if USAR_SEGUNDO_MAPA else "MAPA 1 (Original)"
        print(
            f"{resultado['metodo'].upper()} | {mapa_etiqueta} | "
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
                elif celda in obstaculos_activos:
                    linea += EMOJI_OBSTACULO
                else:
                    linea += EMOJI_LIBRE
            print(linea)
            
        print(
            f"Paso {paso}/{len(trayectoria) - 1} | "
            f"Obs(8bit): {observaciones[paso - 1]:08b} | "
            f"Accion: {acciones[paso - 1]}"
        )
        if pausa:
            time.sleep(pausa)


def mostrar_resultado(resultado, pausa):
    """Muestra la animación y métricas finales."""
    dibujar(resultado, pausa)
    estado = "LLEGO" if resultado["llego"] else "NO LLEGO"
    print(f"\nResultado: {estado}")
    print(f"Pasos útiles: {resultado['pasos_utiles']}")
    print(f"Diversidad final: {resultado['diversidad']}/{TAMANO_POBLACION}")
    if pausa:
        time.sleep(1)


def mostrar_poblacion(resultados, pausa):
    """Anima toda la población resultante."""
    for indice, resultado_actual in enumerate(resultados, start=1):
        print(f"\nIndividuo {indice}/{len(resultados)}")
        dibujar(resultado_actual, pausa)


def main():
    global USAR_SEGUNDO_MAPA
    parser = argparse.ArgumentParser(
        description="Reto de Modificación - Políticas con percepción extendida, memoria y segundo mapa."
    )
    parser.add_argument(
        "--modo",
        choices=("comparar", "mutacion", "cruce"),
        default="comparar",
    )
    parser.add_argument("--semilla", type=int, default=7)
    parser.add_argument("--sin-pausa", action="store_true")
    parser.add_argument(
        "--usar-mapa-2",
        action="store_true",
        help="Activa la segunda distribución de obstáculos (Modificación 4).",
    )
    parser.add_argument(
        "--identificador",
        help="Texto usado en el nombre del archivo de salida.",
    )
    parser.add_argument(
        "--cargar",
        metavar="ARCHIVO",
        help="Carga una generación guardada previamente.",
    )
    parser.add_argument(
        "--ejecutar",
        choices=("mejor", "todos"),
        default="mejor",
    )
    parser.add_argument(
        "--no-guardar",
        action="store_true",
    )
    args = parser.parse_args()

    # Activación de la Modificación 4 mediante argumento CLI si se solicita
    if args.usar_mapa_2:
        USAR_SEGUNDO_MAPA = True

    pausa = 0 if args.sin_pausa else 0.08
    
    if args.cargar:
        datos = cargar_generacion(args.cargar)
        resultados = ejecutar_poblacion(datos, args.ejecutar)
        if args.ejecutar == "todos":
            mostrar_poblacion(resultados, pausa)
        else:
            mostrar_resultado(resultados[0], pausa)
        return

    experimentos = {
        "mutacion": ("solo mutacion", False),
        "cruce": ("mutacion + cruce", True),
    }
    modos = tuple(experimentos) if args.modo == "comparar" else (args.modo,)
    resultados = []

    for indice, modo in enumerate(modos):
        _, usar_cruce = experimentos[modo]
        resultado_actual = evolucionar(
            tasa_mutacion=0.08,
            usar_cruce=usar_cruce,
            semilla=args.semilla + indice,
        )
        resultados.append(resultado_actual)
        if not args.no_guardar:
            identificador = args.identificador or (
                f"{modo}_modificado_semilla{args.semilla + indice}"
            )
            nombre = guardar_generacion(
                resultado_actual, identificador, args.semilla + indice
            )
            print(f"Generación guardada en: {nombre}")
            
        if args.ejecutar == "todos":
            mostrar_poblacion(
                ejecutar_poblacion(
                    {
                        "generacion": resultado_actual["generacion"],
                        "metodo": resultado_actual["metodo"],
                        "poblacion": resultado_actual["poblacion"],
                    },
                    "todos",
                ),
                pausa,
            )
        else:
            mostrar_resultado(resultado_actual, pausa)

    if len(resultados) == 2:
        print("\nCOMPARACION DEL RETO DE MODIFICACION")
        print("Metodo             Llego  Generaciones  Puntaje  Choques  Diversidad")
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