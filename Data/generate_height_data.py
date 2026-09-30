#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador didáctico de datos sintéticos: edad -> estatura.
Este script crea un CSV con dos columnas: `age` (años) y `height` (centímetros).
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame con `n` muestras.

    Parámetros:
    - n: número de muestras a generar.
    - seed: semilla para el generador aleatorio (reproducibilidad).

    Retorna:
    - pd.DataFrame con columnas `age` y `height`.
    """
    rng = np.random.default_rng(seed)

    # 1) Generar edades en años, con dos decimales para simular mediciones reales
    ages = rng.uniform(0, 18, size=n)

    # 2) Generar estaturas con una relación no lineal + ruido
    heights = 45 + 5.2 * ages + 0.08 * (ages ** 2) + rng.normal(0, 3.0, size=n)

    # 3) Empaquetar y redondear para apariencia de datos reales
    df = pd.DataFrame({
        "age": np.round(ages, 2),
        "height": np.round(heights, 2),
    })
    return df


def main() -> None:
    """Interfaz de línea de comandos."""
    p = argparse.ArgumentParser(description="Generador de datos sintéticos edad->estatura")
    p.add_argument("--n", type=int, default=500, help="Número de muestras")
    p.add_argument("--out", type=str, default="data/estatura_ninos.csv", help="Archivo CSV de salida")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    args = p.parse_args()

    # Generar DataFrame
    df = generate(args.n, seed=args.seed)

    # Asegurarse de que la carpeta de salida exista
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Guardar CSV sin índice
    df.to_csv(args.out, index=False)
    print(f"Generado {len(df)} muestras en {args.out}")


if __name__ == "__main__":
    main()