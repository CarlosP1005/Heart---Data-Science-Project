"""Pruebas del pipeline de preparación de datos (`src/pipelines/feature_pipeline`)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from pipelines.feature_pipeline import (
    OBJETIVO,
    construir_pipeline_features,
    limpiar_datos,
)

COLUMNAS = [
    "age",
    "sex",
    "chest_pain",
    "rest_bp",
    "chol",
    "fbs",
    "rest_ecg",
    "max_hr",
    "exang",
    "old_peak",
    "slope",
    "ca",
    "thal",
    OBJETIVO,
]

ATRIBUTOS = [c for c in COLUMNAS if c != OBJETIVO]


def _fila(**cambios: str) -> dict[str, str]:
    """Construye un registro válido (todo texto, como se lee el CSV crudo) y lo modifica."""
    base = {
        "age": "63",
        "sex": "Male",
        "chest_pain": "typical",
        "rest_bp": "145",
        "chol": "233",
        "fbs": "1",
        "rest_ecg": "left ventricular hypertrophy",
        "max_hr": "150",
        "exang": "0",
        "old_peak": "2.3",
        "slope": "3",
        "ca": "0",
        "thal": "fixed",
        OBJETIVO: "0",
    }
    base.update(cambios)
    return base


@pytest.fixture
def datos_crudos() -> pd.DataFrame:
    """Muestra sintética con los mismos defectos del archivo original."""
    filas = [
        _fila(age="63", chol="233"),
        _fila(age="63", chol="233"),  # duplicado exacto
        _fila(age="67", sex="Female", chest_pain="asymptomatic", chol="286", disease="1"),
        _fila(age="41", sex="2345", chol="204", disease="1"),  # categoría corrupta -> NaN
        _fila(age="fggfds", chol="250"),  # numérico corrupto -> NaN
        _fila(age="57", rest_ecg="normal  ", chol="354", thal="normal", disease="1"),
        _fila(age="56", chol="236", disease=""),  # sin target -> se elimina
        _fila(age="44", chest_pain="nonanginal", chol="263", max_hr="173", disease="1"),
        _fila(age="52", chest_pain="atypical", chol="199", thal="reversable", disease="0"),
        _fila(age="59", sex="Female", chol="212", slope="1", ca="2", disease="1"),
    ]
    return pd.DataFrame(filas, columns=COLUMNAS).replace("", None).astype("object")


def test_limpiar_datos_elimina_duplicados_y_filas_sin_target(datos_crudos: pd.DataFrame) -> None:
    """La limpieza descarta duplicados exactos y registros sin variable objetivo."""
    limpios = limpiar_datos(datos_crudos)

    assert len(limpios) < len(datos_crudos)
    assert limpios[OBJETIVO].notna().all()
    assert not limpios.duplicated().any()


def test_limpiar_datos_anula_valores_fuera_de_dominio(datos_crudos: pd.DataFrame) -> None:
    """Los valores corruptos se convierten en faltantes en lugar de propagarse."""
    limpios = limpiar_datos(datos_crudos)

    assert limpios["sex"].dropna().isin({"Male", "Female"}).all()
    assert pd.api.types.is_numeric_dtype(limpios["age"])
    assert limpios["age"].dropna().between(18, 110).all()


def test_limpiar_datos_normaliza_espacios(datos_crudos: pd.DataFrame) -> None:
    """`rest_ecg` con espacios finales sigue siendo una categoría válida tras limpiar."""
    limpios = limpiar_datos(datos_crudos)

    assert "normal" in set(limpios["rest_ecg"].dropna())


def test_pipeline_produce_matriz_numerica_sin_faltantes(datos_crudos: pd.DataFrame) -> None:
    """El pipeline entrega una matriz totalmente numérica, finita y sin faltantes."""
    limpios = limpiar_datos(datos_crudos)
    pipeline = construir_pipeline_features(ATRIBUTOS)

    procesado = pipeline.fit_transform(limpios[ATRIBUTOS], limpios[OBJETIVO])
    matriz = np.asarray(procesado, dtype="float64")

    assert matriz.shape[0] == len(limpios)
    assert matriz.shape[1] > len(ATRIBUTOS)
    assert not np.isnan(matriz).any()
    assert np.isfinite(matriz).all()


def test_pipeline_transforma_test_con_las_mismas_columnas(datos_crudos: pd.DataFrame) -> None:
    """`transform` sobre datos nuevos devuelve exactamente las columnas aprendidas en `fit`."""
    limpios = limpiar_datos(datos_crudos)
    pipeline = construir_pipeline_features(ATRIBUTOS)

    entrenamiento = pipeline.fit_transform(limpios[ATRIBUTOS], limpios[OBJETIVO])
    nuevo = pipeline.transform(limpios[ATRIBUTOS].head(1))

    assert list(np.asarray(pipeline.get_feature_names_out())) == list(entrenamiento.columns)
    assert list(nuevo.columns) == list(entrenamiento.columns)
    assert len(nuevo) == 1


def test_pipeline_es_reajustable_sin_estado_compartido(datos_crudos: pd.DataFrame) -> None:
    """Dos pipelines construidos por la fábrica son independientes entre sí."""
    limpios = limpiar_datos(datos_crudos)

    primero = construir_pipeline_features(ATRIBUTOS)
    segundo = construir_pipeline_features(ATRIBUTOS)
    primero.fit(limpios[ATRIBUTOS], limpios[OBJETIVO])

    with pytest.raises(NotFittedError):
        segundo.transform(limpios[ATRIBUTOS])
