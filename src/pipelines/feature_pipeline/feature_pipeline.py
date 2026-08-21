"""Pipeline de preparación de datos del proyecto *Heart Disease*.

Este módulo consolida el trabajo de la etapa **4 - Feature Engineering**
(`notebooks/4-feat_eng/04_feature_engineering-CJPS-2026-08-20.ipynb`) en un único
punto de verdad importable, para que las etapas siguientes (modelo base, entrenamiento,
inferencia) reutilicen **exactamente** las mismas transformaciones sin copiar código.

Contiene tres bloques:

1. **Contratos de datos**: dominios categóricos, rangos fisiológicos y códigos válidos.
2. **Limpieza determinista** (`limpiar_datos`): no aprende parámetros de los datos, por lo
   que puede aplicarse antes de la partición `train`/`test` sin producir fuga.
3. **Pipeline de atributos** (`construir_pipeline_features`): todo lo que *sí* aprende
   parámetros (imputación, límites de atípicos, escalado, codificación) y que por lo
   tanto debe ajustarse únicamente con `train`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
)

__all__ = [
    "ATRIBUTOS_DERIVADOS",
    "CODIGOS_VALIDOS",
    "DOMINIOS_CATEGORICOS",
    "OBJETIVO",
    "RANGOS_NUMERICOS",
    "BaseIQR",
    "MarcadorAtipicos",
    "RecortadorAtipicos",
    "agregar_atributos_derivados",
    "construir_pipeline_features",
    "limpiar_datos",
    "nombres_derivados",
]

# --------------------------------------------------------------------------------------
# 1. Contratos de datos
# --------------------------------------------------------------------------------------

OBJETIVO = "disease"

#: Valores admitidos en cada columna categórica; cualquier otro es error de captura.
DOMINIOS_CATEGORICOS: dict[str, set[str]] = {
    "sex": {"Male", "Female"},
    "chest_pain": {"typical", "atypical", "nontypical", "nonanginal", "asymptomatic"},
    "rest_ecg": {"normal", "left ventricular hypertrophy", "ST-T wave abnormality"},
    "thal": {"normal", "fixed", "reversable"},
}

#: Rangos fisiológicamente plausibles de las columnas continuas.
RANGOS_NUMERICOS: dict[str, tuple[float, float]] = {
    "age": (18.0, 110.0),
    "rest_bp": (60.0, 260.0),
    "chol": (80.0, 700.0),
    "max_hr": (50.0, 250.0),
    "old_peak": (0.0, 10.0),
}

#: Columnas discretas con un conjunto cerrado de códigos numéricos válidos.
CODIGOS_VALIDOS: dict[str, set[float]] = {
    "fbs": {0.0, 1.0},
    "exang": {0.0, 1.0},
    "slope": {1.0, 2.0, 3.0},
    "ca": {0.0, 1.0, 2.0, 3.0},
    OBJETIVO: {0.0, 1.0},
}

COLUMNAS_NUMERICAS: list[str] = list(RANGOS_NUMERICOS)
COLUMNAS_CODIFICADAS: list[str] = list(CODIGOS_VALIDOS)

#: Proporción máxima de celdas vacías tolerada en una fila.
UMBRAL_FALTANTES_FILA = 0.5

#: Atributos creados por `agregar_atributos_derivados`, en orden.
ATRIBUTOS_DERIVADOS: list[str] = [
    "fc_maxima_teorica",
    "reserva_cardiaca",
    "chol_rel_edad",
    "carga_presion",
    "log_old_peak",
    "sqrt_chol",
    "st_deprimido",
]

# Grupos de columnas por tratamiento. `age`, `chol` y `max_hr` aparecen dos veces a
# propósito: se usan en su versión continua escalada y en su versión discretizada.
CONTINUAS: list[str] = [
    "age",
    "rest_bp",
    "chol",
    "max_hr",
    "old_peak",
    "fc_maxima_teorica",
    "reserva_cardiaca",
    "chol_rel_edad",
    "carga_presion",
    "log_old_peak",
    "sqrt_chol",
]
A_DISCRETIZAR: list[str] = ["age", "chol", "max_hr"]
MARCAR_ATIPICOS: list[str] = ["rest_bp", "chol", "old_peak"]
NOMINALES: list[str] = ["sex", "chest_pain", "rest_ecg", "thal"]
ORDINALES: list[str] = ["slope", "ca"]
BINARIAS: list[str] = ["fbs", "exang", "st_deprimido"]


# --------------------------------------------------------------------------------------
# 2. Limpieza determinista (sin parámetros aprendidos -> sin fuga de datos)
# --------------------------------------------------------------------------------------


def normalizar_texto(datos: pd.DataFrame) -> pd.DataFrame:
    """Quita espacios sobrantes y unifica el espaciado interno de las columnas de texto."""
    datos = datos.copy()
    for columna in DOMINIOS_CATEGORICOS:
        datos[columna] = datos[columna].str.strip().str.replace(r"\s+", " ", regex=True)
    return datos


def tipificar_y_validar(datos: pd.DataFrame) -> pd.DataFrame:
    """Convierte cada columna a su tipo correcto y anula los valores fuera de dominio."""
    datos = datos.copy()
    for columna, dominio in DOMINIOS_CATEGORICOS.items():
        datos[columna] = datos[columna].where(datos[columna].isin(dominio))
    for columna, (limite_inf, limite_sup) in RANGOS_NUMERICOS.items():
        valores = pd.to_numeric(datos[columna], errors="coerce")
        datos[columna] = valores.where(valores.between(limite_inf, limite_sup))
    for columna, codigos in CODIGOS_VALIDOS.items():
        valores = pd.to_numeric(datos[columna], errors="coerce")
        datos[columna] = valores.where(valores.isin(codigos))
    return datos


def eliminar_duplicados_parciales(datos: pd.DataFrame) -> pd.DataFrame:
    """Descarta las filas cuyos valores presentes están contenidos en otra fila más completa.

    Dos registros del mismo paciente pueden diferir sólo en qué celdas quedaron vacías tras
    invalidar los valores corruptos; `drop_duplicates()` no los detecta. Se recorren las filas
    de la más completa a la menos completa y se descarta toda fila "subsumida" por otra.
    """
    orden = datos.notna().sum(axis=1).sort_values(ascending=False, kind="stable").index
    ordenado = datos.loc[orden]
    valores = ordenado.to_numpy(dtype=object)
    presencia = ordenado.notna().to_numpy()

    conservadas: list[int] = []
    for i in range(len(valores)):
        columnas_presentes = np.flatnonzero(presencia[i])
        subsumida = any(
            bool((valores[j][columnas_presentes] == valores[i][columnas_presentes]).all())
            for j in conservadas
        )
        if not subsumida:
            conservadas.append(i)

    indices = ordenado.index[conservadas]
    return datos.loc[datos.index.isin(indices)].reset_index(drop=True)


def limpiar_datos(datos_crudos: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza completa de la etapa 4 al DataFrame crudo leído como texto.

    Pasos: normalizar texto -> tipificar y validar dominios -> descartar filas sin *target*
    -> descartar filas con demasiados faltantes -> quitar duplicados exactos y parciales.
    """
    datos = normalizar_texto(datos_crudos)
    datos = tipificar_y_validar(datos)
    datos = datos.dropna(subset=[OBJETIVO])
    proporcion_faltantes = datos.drop(columns=[OBJETIVO]).isna().mean(axis=1)
    datos = datos[proporcion_faltantes <= UMBRAL_FALTANTES_FILA]
    datos = datos.drop_duplicates(ignore_index=True)
    datos = eliminar_duplicados_parciales(datos)
    datos[OBJETIVO] = datos[OBJETIVO].astype("int8")
    return datos


# --------------------------------------------------------------------------------------
# 3. Transformadores personalizados
# --------------------------------------------------------------------------------------


class BaseIQR(BaseEstimator, TransformerMixin):
    """Clase base: aprende los límites de Tukey (Q1 - k·IQR, Q3 + k·IQR) por columna."""

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BaseIQR:
        """Calcula los límites de Tukey de cada columna usando sólo el conjunto recibido."""
        del y  # la regla es no supervisada
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        rango_intercuartil = q3 - q1
        self.limite_inferior_ = q1 - self.factor * rango_intercuartil
        self.limite_superior_ = q3 + self.factor * rango_intercuartil
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self


class RecortadorAtipicos(BaseIQR):
    """Recorta (winsoriza) los valores extremos al rango aprendido en entrenamiento."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Acota cada columna al intervalo aprendido en `fit`."""
        return X.clip(lower=self.limite_inferior_, upper=self.limite_superior_, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Devuelve los nombres de salida: los mismos de entrada."""
        del input_features
        return self.feature_names_in_


class MarcadorAtipicos(BaseIQR):
    """Crea un indicador binario por columna: 1 si el valor original era atípico."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Genera las columnas `<atributo>_atipico` a partir de los límites aprendidos."""
        fuera_de_rango = X.lt(self.limite_inferior_) | X.gt(self.limite_superior_)
        marcas = fuera_de_rango.astype("float64")
        marcas.columns = [f"{columna}_atipico" for columna in X.columns]
        return marcas

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Devuelve los nombres de salida con el sufijo `_atipico`."""
        del input_features
        return np.asarray([f"{c}_atipico" for c in self.feature_names_in_], dtype=object)


# --------------------------------------------------------------------------------------
# 4. Atributos derivados
# --------------------------------------------------------------------------------------


def agregar_atributos_derivados(X: pd.DataFrame) -> pd.DataFrame:
    """Añade atributos con sentido clínico y transformaciones que linealizan colas sesgadas.

    Los faltantes se propagan a propósito: los imputa el pipeline con estadísticos de `train`.
    """
    # Se reinicia el índice porque `KBinsDiscretizer` devuelve un índice posicional y
    # `ColumnTransformer` exige que todas las ramas compartan el mismo índice al concatenar.
    X = X.copy().reset_index(drop=True)
    X["fc_maxima_teorica"] = 220.0 - X["age"]
    X["reserva_cardiaca"] = X["max_hr"] / X["fc_maxima_teorica"]
    X["chol_rel_edad"] = X["chol"] / X["age"]
    X["carga_presion"] = X["rest_bp"] / X["max_hr"]
    X["log_old_peak"] = np.log1p(X["old_peak"])
    X["sqrt_chol"] = np.sqrt(X["chol"])
    X["st_deprimido"] = np.where(
        X["old_peak"].isna(), np.nan, (X["old_peak"] > 0).astype("float64")
    )
    return X


def nombres_derivados(_transformador: Any, input_features: Any) -> np.ndarray:
    """`get_feature_names_out` del `FunctionTransformer`: originales + derivados."""
    return np.asarray([*list(input_features), *ATRIBUTOS_DERIVADOS], dtype=object)


# --------------------------------------------------------------------------------------
# 5. Ensamblaje del pipeline
# --------------------------------------------------------------------------------------


def construir_pipeline_features(atributos_seleccionados: list[str] | None = None) -> Pipeline:
    """Construye el pipeline **sin ajustar** de la etapa 4.

    Args:
        atributos_seleccionados: columnas de entrada que sobrevivieron a la selección de
            atributos. Si es `None` se usan todas las columnas conocidas del dataset.

    Returns:
        Un `Pipeline` de dos pasos (`atributos_derivados` -> `preprocesamiento`) que va del
        DataFrame limpio a la matriz numérica lista para el estimador.
    """
    if atributos_seleccionados is None:
        atributos_seleccionados = [
            *COLUMNAS_NUMERICAS,
            *DOMINIOS_CATEGORICOS,
            *[c for c in COLUMNAS_CODIFICADAS if c != OBJETIVO],
        ]

    disponibles = set(atributos_seleccionados) | set(ATRIBUTOS_DERIVADOS)
    continuas = [c for c in CONTINUAS if c in disponibles]
    a_discretizar = [c for c in A_DISCRETIZAR if c in disponibles]
    marcar_atipicos = [c for c in MARCAR_ATIPICOS if c in disponibles]
    nominales = [c for c in NOMINALES if c in disponibles]
    ordinales = [c for c in ORDINALES if c in disponibles]
    binarias = [c for c in BINARIAS if c in disponibles]

    generador_atributos = FunctionTransformer(
        agregar_atributos_derivados,
        feature_names_out=nombres_derivados,
        validate=False,
    )

    # Continuas: imputar -> acotar atípicos -> escalar de forma robusta.
    pipeline_continuas = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("recorte_atipicos", RecortadorAtipicos(factor=1.5)),
            ("escalado", RobustScaler()),
        ]
    )
    # Discretización: imputar -> cuartiles -> one-hot de los intervalos.
    pipeline_discretizacion = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            (
                "discretizacion",
                KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile"),
            ),
        ]
    )
    # Marcado de atípicos: imputar -> indicadores binarios.
    pipeline_marcas = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("marcado_atipicos", MarcadorAtipicos(factor=1.5)),
        ]
    )
    # Nominales: imputar con la moda -> One-Hot Encoding.
    pipeline_nominales = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            (
                "codificacion",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                    min_frequency=0.01,
                ),
            ),
        ]
    )
    # Ordinales: imputar con la moda -> Ordinal Encoding con el orden declarado.
    pipeline_ordinales = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            (
                "codificacion",
                OrdinalEncoder(
                    categories=[sorted(CODIGOS_VALIDOS[c]) for c in ordinales],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    # Binarias: sólo imputar con la moda (ya vienen codificadas 0/1).
    pipeline_binarias = Pipeline(steps=[("imputacion", SimpleImputer(strategy="most_frequent"))])

    preprocesador = ColumnTransformer(
        transformers=[
            ("continuas", pipeline_continuas, continuas),
            ("discretizadas", pipeline_discretizacion, a_discretizar),
            ("marcas_atipicos", pipeline_marcas, marcar_atipicos),
            ("nominales", pipeline_nominales, nominales),
            ("ordinales", pipeline_ordinales, ordinales),
            ("binarias", pipeline_binarias, binarias),
        ],
        remainder="drop",  # la selección de atributos se materializa aquí
        verbose_feature_names_out=True,
    )

    pipeline = Pipeline(
        steps=[
            ("atributos_derivados", generador_atributos),
            ("preprocesamiento", preprocesador),
        ]
    )
    # Salida como DataFrame en todos los pasos: conserva los nombres de las columnas a lo
    # largo del pipeline y hace el resultado auditable, sin depender de la configuración
    # global de scikit-learn (`sklearn.set_config`).
    pipeline.set_output(transform="pandas")
    return pipeline
