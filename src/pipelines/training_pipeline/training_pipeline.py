"""Pipeline de entrenamiento del modelo seleccionado en la etapa 6.

Este módulo es el resultado ejecutable de
`notebooks/5-models/06_seleccion_modelo_automl-CJPS-2026-08-21.ipynb`, donde se compararon
seis familias de modelos propuestas por dos motores de AutoML (FLAML y MLJar) contra el
modelo base de la etapa 5.

El modelo ganador fue una **regresión logística con regularización L2 fuerte**. Sus
hiperparámetros no se escogieron a mano: son los que encontró la búsqueda de FLAML y se
fijan aquí como constantes para que el entrenamiento sea reproducible sin necesidad de
volver a lanzar la búsqueda —que es cara— cada vez que se reentrena.

El módulo expone tres operaciones y nada más:

- `construir_pipeline_modelo`: arma el `Pipeline` sin ajustar (preprocesamiento + modelo).
- `entrenar`: limpia el DataFrame crudo y ajusta ese pipeline.
- `guardar_artefacto` / `cargar_artefacto`: serializan y recuperan pipeline + metadatos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipelines.feature_pipeline import OBJETIVO, construir_pipeline_features, limpiar_datos

__all__ = [
    "ATRIBUTOS_ENTRADA",
    "HIPERPARAMETROS",
    "NOMBRE_ARTEFACTO",
    "SEMILLA",
    "UMBRAL_RECOMENDADO",
    "cargar_artefacto",
    "construir_pipeline_modelo",
    "entrenar",
    "guardar_artefacto",
    "predecir",
]

#: Semilla usada en toda la etapa 6 (y en las etapas 4 y 5).
SEMILLA = 42

#: Atributos de entrada que sobrevivieron a la selección de la etapa 4 (`fbs` se descartó).
ATRIBUTOS_ENTRADA: list[str] = [
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "rest_ecg",
    "max_hr",
    "exang",
    "old_peak",
    "slope",
    "ca",
    "thal",
]

#: Hiperparámetros encontrados por FLAML (200 configuraciones, métrica ROC-AUC).
HIPERPARAMETROS: dict[str, float] = {"C": 0.033001297334653665}

#: Umbral que maximiza F2 sobre el conjunto de entrenamiento. El 0.5 por defecto supone
#: que un falso negativo y un falso positivo cuestan lo mismo, que aquí no es cierto.
UMBRAL_RECOMENDADO = 0.36

#: Nombre del artefacto serializado dentro de `models/`.
NOMBRE_ARTEFACTO = "modelo_seleccionado_automl.joblib"


def construir_modelo() -> LogisticRegression:
    """Crea el estimador ganador con los hiperparámetros que encontró el AutoML."""
    return LogisticRegression(
        C=HIPERPARAMETROS["C"],
        max_iter=5000,
        random_state=SEMILLA,
    )


def construir_pipeline_modelo(atributos: list[str] | None = None) -> Pipeline:
    """Devuelve el pipeline completo **sin ajustar**: preprocesamiento + modelo.

    Args:
        atributos: columnas de entrada. Si es `None` se usan `ATRIBUTOS_ENTRADA`.

    Returns:
        Un `Pipeline` que va del DataFrame limpio de columnas clínicas a la predicción.
        Como no está ajustado, puede pasarse directamente a `cross_validate` sin producir
        fuga: el preprocesamiento se re-aprende dentro de cada *fold*.
    """
    seleccion = ATRIBUTOS_ENTRADA if atributos is None else atributos
    return Pipeline(
        steps=[
            ("features", construir_pipeline_features(seleccion)),
            ("modelo", construir_modelo()),
        ]
    )


def entrenar(datos_crudos: pd.DataFrame, atributos: list[str] | None = None) -> Pipeline:
    """Limpia el DataFrame crudo y ajusta el pipeline completo sobre todos los registros.

    Args:
        datos_crudos: contenido de `corazon.csv` leído con `dtype=str`.
        atributos: columnas de entrada; `None` usa `ATRIBUTOS_ENTRADA`.

    Returns:
        El `Pipeline` ya ajustado, listo para `predict` / `predict_proba`.
    """
    seleccion = ATRIBUTOS_ENTRADA if atributos is None else atributos
    datos = limpiar_datos(datos_crudos)
    pipeline = construir_pipeline_modelo(seleccion)
    pipeline.fit(datos[seleccion], datos[OBJETIVO])
    return pipeline


def predecir(
    pipeline: Pipeline, datos: pd.DataFrame, umbral: float = UMBRAL_RECOMENDADO
) -> pd.DataFrame:
    """Aplica el pipeline y devuelve probabilidad y decisión para cada paciente.

    Args:
        pipeline: pipeline ya ajustado.
        datos: DataFrame **ya limpio** (salida de `limpiar_datos`) con las columnas de
            `ATRIBUTOS_ENTRADA`. El pipeline no acepta el CSV crudo leído como texto: la
            limpieza es un paso previo y deliberadamente separado.
        umbral: probabilidad a partir de la cual se declara enfermo al paciente.

    Returns:
        DataFrame con las columnas `probabilidad_enfermedad` y `prediccion`.
    """
    probabilidades = np.asarray(pipeline.predict_proba(datos))[:, 1]
    return pd.DataFrame(
        {
            "probabilidad_enfermedad": probabilidades,
            "prediccion": (probabilidades >= umbral).astype("int8"),
        },
        index=datos.index,
    )


def guardar_artefacto(
    pipeline: Pipeline, ruta: Path, metadatos: dict[str, Any] | None = None
) -> Path:
    """Serializa el pipeline junto con sus metadatos en un único archivo `.joblib`.

    Se guarda un diccionario y no el pipeline pelado a propósito: un artefacto que no
    declara con qué datos, semilla y versión de scikit-learn fue entrenado no es auditable.

    Args:
        pipeline: pipeline ajustado.
        ruta: destino del archivo `.joblib`.
        metadatos: información adicional a incrustar.

    Returns:
        La ruta escrita.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contenido: dict[str, Any] = {
        "pipeline": pipeline,
        "metadatos": {
            "modelo": "LogisticRegression (L2)",
            "motor_automl": "FLAML",
            "hiperparametros": HIPERPARAMETROS,
            "atributos_entrada": ATRIBUTOS_ENTRADA,
            "umbral_recomendado": UMBRAL_RECOMENDADO,
            "semilla": SEMILLA,
            **(metadatos or {}),
        },
    }
    joblib.dump(contenido, ruta)
    return ruta


def cargar_artefacto(ruta: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Recupera el pipeline y sus metadatos desde un archivo `.joblib`.

    Args:
        ruta: archivo generado por `guardar_artefacto`.

    Returns:
        Una tupla `(pipeline, metadatos)`.

    Raises:
        KeyError: si el archivo no tiene la estructura que produce `guardar_artefacto`.
    """
    contenido = joblib.load(ruta)
    if "pipeline" not in contenido or "metadatos" not in contenido:
        raise KeyError("El artefacto no contiene las claves 'pipeline' y 'metadatos'.")
    return contenido["pipeline"], contenido["metadatos"]
