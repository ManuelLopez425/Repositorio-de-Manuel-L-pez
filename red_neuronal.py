"""
================================================================================
  RED NEURONAL SIMPLE CON PYTORCH - PREDICCIÓN DE DEMANDA DE COMIDA
================================================================================

¿POR QUÉ UNA RED NEURONAL?
  Un modelo tradicional (como regresión lineal) asume que la relación entre
  precio y pedidos es una línea recta. Pero la realidad es más compleja:
  una promoción combinada con precio bajo puede disparar la demanda de forma
  no lineal. Las redes neuronales capturan esas relaciones complejas.

¿POR QUÉ PYTORCH Y NO TENSORFLOW?
  PyTorch tiene una sintaxis más cercana a Python puro, lo que lo hace más
  fácil de entender y depurar. Es el estándar en investigación académica.

CÓMO ADAPTAR A OTRO ARCHIVO:
  1. Cambia ARCHIVO_CSV al nombre de tu archivo
  2. Cambia COLUMNAS_ENTRADA a las columnas que quieres usar como entrada
  3. Cambia COLUMNA_OBJETIVO a la columna que quieres predecir

REQUISITOS:
  pip install torch pandas scikit-learn matplotlib
================================================================================
"""

# ¿POR QUÉ pandas?
#   Es la librería estándar para leer y manipular tablas de datos en Python.
#   Convierte el CSV en un objeto "DataFrame", como una hoja de cálculo en memoria.
import pandas as pd

# ¿POR QUÉ torch?
#   Es el núcleo de PyTorch. Maneja "tensores" (arrays multidimensionales que
#   pueden calcular gradientes automáticamente, lo cual es esencial para entrenar redes).
import torch

# ¿POR QUÉ torch.nn?
#   Es el submódulo de "neural networks". Contiene capas (Linear), funciones de
#   activación (ReLU), funciones de pérdida (MSELoss), etc. Es la caja de herramientas
#   para construir la arquitectura de la red.
import torch.nn as nn

# ¿POR QUÉ DataLoader y TensorDataset?
#   TensorDataset: agrupa X e y en pares (input, output) para que no se desordenen.
#   DataLoader: divide esos pares en "lotes" (batches) y los mezcla aleatoriamente
#   en cada época. Entrenar en lotes es más eficiente que hacerlo fila por fila.
from torch.utils.data import DataLoader, TensorDataset

# ¿POR QUÉ train_test_split?
#   Necesitamos datos que la red NUNCA ha visto para evaluar si realmente aprendió
#   o si solo memorizó los datos de entrenamiento (overfitting). Esta función
#   separa el dataset en entrenamiento (80%) y prueba (20%) de forma aleatoria.
from sklearn.model_selection import train_test_split

# ¿POR QUÉ StandardScaler?
#   Las redes neuronales son sensibles a la escala de los datos.
#   Si "checkout_price" vale ~300 y "emailer_for_promotion" vale 0 o 1,
#   la red le daría mucho más peso al precio solo por ser un número más grande,
#   aunque la promoción sea igual o más importante.
#   StandardScaler lleva todos los valores a la misma escala (media=0, desviación=1).
from sklearn.preprocessing import StandardScaler

# ¿POR QUÉ estas métricas?
#   mean_absolute_error (MAE): dice en promedio cuántos pedidos se equivoca la red.
#     Es fácil de interpretar: "me equivoco ±150 pedidos".
#   r2_score (R²): mide qué porcentaje de la variación en los pedidos explica el modelo.
#     1.0 = perfecto, 0.0 = no aprende nada, negativo = peor que adivinar el promedio.
from sklearn.metrics import mean_absolute_error, r2_score

import numpy as np  # Para operaciones numéricas básicas
from pathlib import Path


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN
# (modifica aquí para usar otro archivo o columnas)
# ==============================================================================

CARPETA_SCRIPT = Path(__file__).resolve().parent
ARCHIVO_CSV = CARPETA_SCRIPT / "AUTOMPG.csv"
ARCHIVO_HISTORIAL = CARPETA_SCRIPT / "historial_entrenamiento.csv"

# ¿POR QUÉ estas columnas de entrada?
#   - checkout_price y base_price: el precio afecta directamente la demanda
#     (precio más bajo → más pedidos, generalmente)
#   - emailer_for_promotion: una campaña de email puede disparar los pedidos
#   - homepage_featured: aparecer en la portada da mucha visibilidad
#   - week: puede haber patrones de temporada (semanas festivas, vacaciones)
COLUMNAS_ENTRADA = [
    "displacement",
    "cylinders",
    "weight",
    "acceleration",
    "model_year"
]

COLUMNA_OBJETIVO = "mpg"  # Lo que queremos predecir

# ¿POR QUÉ 100 épocas?
#   Una época = la red ve TODOS los datos una vez.
#   Pocas épocas → la red no aprende suficiente (underfitting).
#   Demasiadas épocas → puede memorizar los datos (overfitting).
#   100 es un buen punto de partida para datasets medianos.
EPOCAS = 2000

# ¿POR QUÉ tasa de aprendizaje 0.001?
#   Controla qué tan grandes son los pasos al ajustar los pesos.
#   0.1 → pasos grandes, aprende rápido pero puede "saltar" la solución óptima.
#   0.0001 → pasos pequeños, muy lento pero estable.
#   0.001 es el valor por defecto recomendado para el optimizador Adam.
TASA_APRENDIZAJE = 0.1

# ¿POR QUÉ lotes de 64?
#   Lote de 1 → muy lento, mucho ruido en el aprendizaje.
#   Lote de todos los datos → usa mucha memoria, puede atascarse en mínimos locales.
#   64 es un valor clásico: buen balance entre velocidad y estabilidad.
TAMANO_LOTE = 64

# ¿POR QUÉ 20% para prueba?
#   Es la división más común en machine learning.
#   Con 2000 filas: ~1600 para entrenar y ~400 para evaluar. Es suficiente.
PORCENTAJE_PRUEBA = 0.2

# La validación permite observar el aprendizaje; el test queda reservado
# para la evaluación final del modelo.
PORCENTAJE_VALIDACION = 0.2


# ==============================================================================
# SECCIÓN 2: CARGAR Y PREPARAR LOS DATOS
# ==============================================================================

print("=" * 60)
print("  CARGANDO DATOS...")
print("=" * 60)

datos = pd.read_csv(ARCHIVO_CSV)

print(f"  Filas totales:          {len(datos)}")
print(f"  Columnas de entrada:    {COLUMNAS_ENTRADA}")
print(f"  Columna objetivo:       {COLUMNA_OBJETIVO}")
print(f"  Pedidos promedio reales: {datos[COLUMNA_OBJETIVO].mean():.0f}")
print(f"  Pedidos mín / máx:      {datos[COLUMNA_OBJETIVO].min()} / {datos[COLUMNA_OBJETIVO].max()}")
print()

# .values convierte el DataFrame de pandas a un array de numpy
# Las redes trabajan mejor con arrays numéricos puros que con DataFrames
X = datos[COLUMNAS_ENTRADA].values
y = datos[COLUMNA_OBJETIVO].values

# random_state=42: número semilla para el generador aleatorio.
# Garantiza que la división sea siempre la misma, haciendo el experimento reproducible.
X_entrenamiento_validacion, X_prueba, y_entrenamiento_validacion, y_prueba = train_test_split(
    X, y, test_size=PORCENTAJE_PRUEBA, random_state=42
)

X_entren, X_validacion, y_entren, y_validacion = train_test_split(
  X_entrenamiento_validacion,
  y_entrenamiento_validacion,
  test_size=PORCENTAJE_VALIDACION,
  random_state=42
)

# ¿POR QUÉ fit_transform solo en entrenamiento y transform en prueba?
#   fit_transform: APRENDE la media y desviación estándar de los datos de entrenamiento
#                  Y luego transforma esos datos.
#   transform:     Usa la media y desviación que YA aprendió para transformar la prueba.
#   Si hiciéramos fit_transform en la prueba también, estaríamos "filtrando" información
#   del futuro al modelo (data leakage), lo cual hace la evaluación engañosa.
escalador = StandardScaler()
X_entren = escalador.fit_transform(X_entren)
X_validacion = escalador.transform(X_validacion)
X_prueba  = escalador.transform(X_prueba)

print(f"  Datos de entrenamiento: {len(X_entren)} filas")
print(f"  Datos de validación:    {len(X_validacion)} filas")
print(f"  Datos de prueba:        {len(X_prueba)} filas")
print()


# ==============================================================================
# SECCIÓN 3: CONVERTIR DATOS A TENSORES DE PYTORCH
# ==============================================================================

# ¿POR QUÉ tensores y no arrays de numpy directamente?
#   Los tensores de PyTorch son como arrays de numpy, pero tienen una capacidad
#   extra: rastrean automáticamente las operaciones matemáticas para poder calcular
#   gradientes (backpropagation). Sin esto, la red no podría aprender.

# dtype=float32: las redes neuronales usan números de 32 bits (no 64) por defecto.
#   Esto reduce el uso de memoria a la mitad sin perder precisión significativa.
X_entren_tensor = torch.tensor(X_entren, dtype=torch.float32)
y_entren_tensor = torch.tensor(y_entren, dtype=torch.float32).unsqueeze(1)
# ¿POR QUÉ unsqueeze(1)?
#   y_entren tiene forma (1600,) → un vector plano.
#   La red espera (1600, 1) → una columna. unsqueeze(1) agrega esa dimensión extra.

X_prueba_tensor  = torch.tensor(X_prueba,  dtype=torch.float32)
y_prueba_tensor  = torch.tensor(y_prueba,  dtype=torch.float32).unsqueeze(1)
X_validacion_tensor = torch.tensor(X_validacion, dtype=torch.float32)
y_validacion_tensor = torch.tensor(y_validacion, dtype=torch.float32).unsqueeze(1)

# ¿POR QUÉ shuffle=True en el DataLoader?
#   Si los datos están ordenados (por semana, por centro), la red podría aprender
#   el orden en lugar del patrón real. Mezclarlos en cada época evita este sesgo.
cargador_entren = DataLoader(
    TensorDataset(X_entren_tensor, y_entren_tensor),
    batch_size=TAMANO_LOTE,
    shuffle=True
)


# ==============================================================================
# SECCIÓN 4: DEFINIR LA ARQUITECTURA DE LA RED NEURONAL
# ==============================================================================

class RedNeuronal(nn.Module):
    """
    Red neuronal con 2 capas ocultas para predecir un valor numérico.

    ¿POR QUÉ heredar de nn.Module?
      Es el contrato de PyTorch: al heredar de nn.Module, nuestra clase
      automáticamente tiene acceso a métodos como .parameters() (para el
      optimizador), .train() y .eval() (para cambiar modos), etc.

    ARQUITECTURA:
      Entrada (5) → [Capa 1: 16 neuronas + ReLU] → [Capa 2: 8 neuronas + ReLU] → Salida (1)

    ¿POR QUÉ 16 y 8 neuronas?
      Regla general: empezar con pocas neuronas para un dataset pequeño.
      16 → 8 forma un "embudo" que va extrayendo patrones cada vez más abstractos.
      Con más neuronas la red podría memorizar el dataset (overfitting).

    ¿POR QUÉ 2 capas ocultas?
      Una sola capa puede aprender relaciones simples.
      Dos capas permiten aprender combinaciones de esas relaciones (patrones más ricos).
      Para este dataset no necesitamos más profundidad.
    """

    def __init__(self, num_entradas):
        super(RedNeuronal, self).__init__()  # Inicializa la clase padre nn.Module

        # ¿POR QUÉ nn.Sequential?
        #   Agrupa la transformación lineal y la activación en un solo bloque.
        #   Así queda claro que ReLU es PARTE de la capa, no un paso separado.
        #   También evita errores de olvidarse aplicar ReLU en el forward.
        #
        # ¿POR QUÉ nn.Linear?
        #   nn.Linear(entrada, salida) implementa: salida = entrada × peso + sesgo
        #   Es la operación fundamental de cada capa de una red neuronal.
        #
        # ¿POR QUÉ ReLU (Rectified Linear Unit)?
        #   Sin funciones de activación, apilar capas lineales sigue siendo lineal.
        #   ReLU hace: si x < 0 → devuelve 0; si x >= 0 → devuelve x.
        #   Esto introduce no-linealidad, permitiendo a la red aprender
        #   relaciones complejas (curvas, interacciones entre variables).
        self.capa1 = nn.Sequential(nn.Linear(num_entradas, 16), nn.ReLU())  # 5 → 16 + ReLU
        self.capa2 = nn.Sequential(nn.Linear(16, 8),            nn.ReLU())  # 16 → 8 + ReLU
        self.salida = nn.Linear(8, 1)  # 8 → 1, SIN ReLU (es regresión, no clasificación)

    def forward(self, x):
        """
        Define el "camino hacia adelante" de los datos.
        ¿POR QUÉ forward?
          PyTorch llama automáticamente a este método cuando haces red(X).
          Aquí defines cómo los datos viajan de entrada a salida.
        """
        x = self.capa1(x)   # Capa 1: Linear + ReLU (ya incluidos en Sequential)
        x = self.capa2(x)   # Capa 2: Linear + ReLU (ya incluidos en Sequential)
        x = self.salida(x)  # Salida:  solo Linear, SIN activación
        # ¿POR QUÉ sin activación en la salida?
        #   Porque es REGRESIÓN: queremos predecir cualquier número (ej: 300 pedidos).
        #   Sigmoid limitaría la salida entre 0 y 1 → inútil para pedidos.
        #   ReLU cortaría negativos → innecesario aquí.
        return x


num_entradas = len(COLUMNAS_ENTRADA)
red = RedNeuronal(num_entradas)

print(f"  Arquitectura: {num_entradas} entradas → 16 → 8 → 1 salida")
print(f"  Parámetros totales a aprender: {sum(p.numel() for p in red.parameters())}")
print()


# ==============================================================================
# SECCIÓN 5: FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# ==============================================================================

# ¿POR QUÉ MSELoss (Error Cuadrático Medio)?
#   Fórmula: promedio de (predicción - valor_real)²
#   El cuadrado penaliza más los errores grandes que los pequeños.
#   Es la pérdida estándar para problemas de regresión (predecir un número).
#   Si predecimos 500 pedidos cuando eran 200, penaliza más que si predecimos 210.
perdida_fn = nn.MSELoss()

# ¿POR QUÉ Adam y no SGD u otros?
#   Adam (Adaptive Moment Estimation) ajusta automáticamente la tasa de aprendizaje
#   para cada parámetro individualmente, basándose en el historial de gradientes.
#   SGD (Gradiente Estocástico) usa la misma tasa para todos y es más difícil de tunear.
#   Adam converge más rápido y es más robusto a la elección de hiperparámetros.
#   Es la opción por defecto en la mayoría de proyectos modernos de deep learning.
optimizador = torch.optim.Adam(red.parameters(), lr=TASA_APRENDIZAJE)


# ==============================================================================
# SECCIÓN 6: BUCLE DE ENTRENAMIENTO
# ==============================================================================

print("=" * 60)
print("  ENTRENANDO LA RED...")
print("=" * 60)

historial_entrenamiento = []

for epoca in range(EPOCAS):

    # ¿POR QUÉ red.train()?
    #   Activa el modo entrenamiento. En redes más complejas, capas como Dropout
    #   (que apaga neuronas aleatoriamente para evitar overfitting) o BatchNorm
    #   se comportan diferente en entrenamiento vs evaluación. Aquí no las usamos,
    #   pero es buena práctica siempre incluirlo.
    red.train()

    perdida_total = 0.0

    # Cada iteración del for da un "lote" de TAMANO_LOTE filas
    for X_lote, y_lote in cargador_entren:

        # PASO 1: FORWARD PASS — La red hace predicciones con los pesos actuales
        prediccion = red(X_lote)

        # PASO 2: CALCULAR PÉRDIDA — Qué tan lejos estuvo la predicción
        perdida = perdida_fn(prediccion, y_lote)

        # PASO 3: LIMPIAR GRADIENTES
        # ¿POR QUÉ zero_grad()?
        #   PyTorch ACUMULA los gradientes en cada llamada a .backward().
        #   Si no los limpiamos antes de cada lote, se sumarían a los del lote anterior,
        #   haciendo que el aprendizaje sea incorrecto.
        optimizador.zero_grad()

        # PASO 4: BACKWARD PASS — Calcular cómo cambiar cada peso
        # ¿POR QUÉ backward()?
        #   Usa la regla de la cadena (cálculo diferencial) para calcular el gradiente
        #   de la pérdida respecto a CADA peso de la red. Esto dice:
        #   "si aumento este peso un poquito, ¿la pérdida sube o baja, y cuánto?"
        perdida.backward()

        # PASO 5: ACTUALIZAR PESOS
        # ¿POR QUÉ step()?
        #   Aplica el cambio calculado en backward() a todos los pesos.
        #   Cada peso se mueve un poco en la dirección que reduce la pérdida.
        #   Así la red "aprende" de sus errores.
        optimizador.step()

        # Ponderamos por el número de filas: el último lote puede ser más pequeño.
        perdida_total += perdida.item() * len(X_lote)

    perdida_entrenamiento = perdida_total / len(X_entren_tensor)

    # La validación mide generalización; no calcula gradientes ni cambia los pesos.
    red.eval()
    with torch.no_grad():
        prediccion_validacion = red(X_validacion_tensor)
        perdida_validacion = perdida_fn(prediccion_validacion, y_validacion_tensor).item()

    historial_entrenamiento.append({
        "epoca": epoca + 1,
        "train_mse": perdida_entrenamiento,
        "validation_mse": perdida_validacion,
        "learning_rate": optimizador.param_groups[0]["lr"]
    })

    if (epoca + 1) % 10 == 0:
        print(
            f"  Época [{epoca + 1:3d}/{EPOCAS}]  |  "
            f"MSE entrenamiento: {perdida_entrenamiento:,.0f}  |  "
            f"MSE validación: {perdida_validacion:,.0f}"
        )

pd.DataFrame(historial_entrenamiento).to_csv(ARCHIVO_HISTORIAL, index=False)
print(f"  Historial de aprendizaje guardado en: {ARCHIVO_HISTORIAL}")

print()


# ==============================================================================
# SECCIÓN 7: EVALUACIÓN
# ==============================================================================

print("=" * 60)
print("  EVALUANDO EL MODELO EN DATOS NO VISTOS...")
print("=" * 60)

# ¿POR QUÉ red.eval() y torch.no_grad()?
#   red.eval(): desactiva capas como Dropout que se comportan diferente en prueba.
#   torch.no_grad(): le dice a PyTorch que NO calcule gradientes. Como no estamos
#   entrenando, no los necesitamos. Esto ahorra memoria y hace el proceso más rápido.
red.eval()
with torch.no_grad():
    predicciones = red(X_prueba_tensor).numpy()
    reales       = y_prueba_tensor.numpy()

mae = mean_absolute_error(reales, predicciones)
r2  = r2_score(reales, predicciones)

print(f"  Error Absoluto Medio (MAE): {mae:.1f} pedidos")
print(f"  → La red se equivoca ±{mae:.0f} pedidos en promedio")
print()
print(f"  R² (coeficiente de determinación): {r2:.3f}")
if r2 > 0.7:
    print(f"  → Buen modelo: explica el {r2*100:.0f}% de la variación en pedidos")
elif r2 > 0.4:
    print(f"  → Modelo moderado: explica el {r2*100:.0f}% de la variación")
else:
    print(f"  → Modelo débil: solo explica el {r2*100:.0f}%. Prueba más épocas o más columnas.")
print()


# ==============================================================================
# SECCIÓN 8: EJEMPLOS DE PREDICCIÓN
# ==============================================================================

print("=" * 60)
print("  EJEMPLOS: PREDICCIÓN vs VALOR REAL")
print("=" * 60)
print(f"  {'#':<4} {'Real':>10} {'Predicción':>12} {'Error':>10}")
print(f"  {'-'*40}")

for i in range(30):
    real  = int(reales[i][0])
    pred  = int(predicciones[i][0])
    error = abs(real - pred)
    print(f"  {i+1:<4} {real:>10} {pred:>12} {error:>10}")

print()
print("=" * 60)
print("  LISTO. Para usar otro CSV:")
print("  1. ARCHIVO_CSV       = 'tu_archivo.csv'")
print("  2. COLUMNAS_ENTRADA  = ['col1', 'col2', ...]")
print("  3. COLUMNA_OBJETIVO  = 'columna_a_predecir'")
print("=" * 60)
