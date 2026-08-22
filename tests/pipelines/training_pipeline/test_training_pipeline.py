"""Pruebas del pipeline de entrenamiento (`src/pipelines/training_pipeline`)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from pipelines.feature_pipeline import OBJETIVO, limpiar_datos
from pipelines.training_pipeline import (
    ATRIBUTOS_ENTRADA,
    HIPERPARAMETROS,
    UMBRAL_RECOMENDADO,
    cargar_artefacto,
    construir_pipeline_modelo,
    entrenar,
    guardar_artefacto,
    predecir,
)

N_MUESTRAS = 60
AUC_MINIMA_ESPERADA = 0.8


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
    """Muestra sintética con dos grupos separables y algunos defectos del archivo real."""
    generador = np.random.default_rng(0)
    filas = []
    for indice in range(N_MUESTRAS):
        enfermo = indice % 2 == 0
        filas.append(
            _fila(
                age=str(40 + int(generador.integers(0, 30))),
                sex="Male" if enfermo else "Female",
                chest_pain="asymptomatic" if enfermo else "nonanginal",
                rest_bp=str(110 + int(generador.integers(0, 50))),
                chol=str(180 + int(generador.integers(0, 120))),
                max_hr=str(110 + int(generador.integers(0, 30)) if enfermo else 160),
                exang="1" if enfermo else "0",
                old_peak="2.4" if enfermo else "0.2",
                slope="2" if enfermo else "1",
                ca=str(int(generador.integers(1, 4))) if enfermo else "0",
                thal="reversable" if enfermo else "normal",
                disease="1" if enfermo else "0",
            )
        )
    filas.append(_fila(age="fggfds", sex="2345", disease="1"))  # basura -> se limpia
    filas.append(_fila(disease="sf"))  # sin target válido -> se elimina
    return pd.DataFrame(filas, dtype=str)


def test_pipeline_sin_ajustar_no_predice() -> None:
    """El pipeline recién construido no está ajustado y debe negarse a predecir."""
    pipeline = construir_pipeline_modelo()
    with pytest.raises(NotFittedError):
        pipeline.predict(pd.DataFrame([{c: 1 for c in ATRIBUTOS_ENTRADA}]))


def test_pipeline_tiene_los_dos_pasos() -> None:
    """El artefacto debe ser preprocesamiento + modelo, no sólo el estimador."""
    pipeline = construir_pipeline_modelo()
    assert list(pipeline.named_steps) == ["features", "modelo"]
    assert pytest.approx(HIPERPARAMETROS["C"]) == pipeline.named_steps["modelo"].C


def test_entrenar_produce_un_pipeline_utilizable(datos_crudos: pd.DataFrame) -> None:
    """`entrenar` va del CSV crudo a un pipeline que devuelve probabilidades válidas."""
    pipeline = entrenar(datos_crudos)
    limpios = limpiar_datos(datos_crudos)
    probabilidades = pipeline.predict_proba(limpios[ATRIBUTOS_ENTRADA].head(5))
    assert probabilidades.shape == (5, 2)
    assert np.all((probabilidades >= 0) & (probabilidades <= 1))
    assert np.allclose(probabilidades.sum(axis=1), 1.0)


def test_entrenar_aprende_algo(datos_crudos: pd.DataFrame) -> None:
    """Sobre datos separables el modelo debe superar claramente al azar."""
    pipeline = entrenar(datos_crudos)
    limpios = limpiar_datos(datos_crudos)
    puntajes = pipeline.predict_proba(limpios[ATRIBUTOS_ENTRADA])[:, 1]
    assert roc_auc_score(limpios[OBJETIVO], puntajes) > AUC_MINIMA_ESPERADA


def test_predecir_respeta_el_umbral(datos_crudos: pd.DataFrame) -> None:
    """La decisión debe salir de comparar la probabilidad contra el umbral recibido."""
    pipeline = entrenar(datos_crudos)
    muestra = limpiar_datos(datos_crudos)[ATRIBUTOS_ENTRADA].head(20)

    resultado = predecir(pipeline, muestra, umbral=UMBRAL_RECOMENDADO)
    assert list(resultado.columns) == ["probabilidad_enfermedad", "prediccion"]
    esperado = (resultado["probabilidad_enfermedad"] >= UMBRAL_RECOMENDADO).astype("int8")
    assert resultado["prediccion"].equals(esperado)

    assert predecir(pipeline, muestra, umbral=0.0)["prediccion"].sum() == len(muestra)
    assert predecir(pipeline, muestra, umbral=1.01)["prediccion"].sum() == 0


def test_artefacto_ida_y_vuelta(datos_crudos: pd.DataFrame, tmp_path: Path) -> None:
    """Guardar y recargar debe reproducir exactamente las mismas probabilidades."""
    pipeline = entrenar(datos_crudos)
    muestra = limpiar_datos(datos_crudos)[ATRIBUTOS_ENTRADA].head(10)
    antes = pipeline.predict_proba(muestra)[:, 1]

    ruta = guardar_artefacto(pipeline, tmp_path / "modelo.joblib", {"prueba": True})
    assert ruta.exists()

    recargado, metadatos = cargar_artefacto(ruta)
    assert np.allclose(recargado.predict_proba(muestra)[:, 1], antes)
    assert metadatos["atributos_entrada"] == ATRIBUTOS_ENTRADA
    assert metadatos["umbral_recomendado"] == UMBRAL_RECOMENDADO
    assert metadatos["prueba"] is True


def test_cargar_artefacto_rechaza_un_archivo_ajeno(tmp_path: Path) -> None:
    """Un `.joblib` que no tenga la estructura esperada debe fallar de forma explícita."""
    ruta = tmp_path / "ajeno.joblib"
    joblib.dump({"otra_cosa": 1}, ruta)
    with pytest.raises(KeyError):
        cargar_artefacto(ruta)
