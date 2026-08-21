# Feature Engineering: pipelines de scikit-learn para el dataset de corazón

## ✨ Context

Cuarta etapa del ciclo de vida del proyecto (`notebooks/4-feat_eng`), siguiendo la guía del
curso [Ciencia de Datos en Producción — Feature Engineering](https://joserzapata.github.io/post/ciencia-datos-proyecto-python/4-feat_eng/).

Tras el análisis exploratorio (univariable, bivariable y multivariable), los datos crudos
de `notebooks/1-data/corazon.csv` siguen sin ser utilizables para entrenar: contienen
duplicados masivos, valores corruptos, faltantes y escalas heterogéneas. Este PR entrega
el notebook que los deja listos para la etapa de modelado, con **todas** las
transformaciones implementadas como `Pipeline` / `ColumnTransformer` de scikit-learn.

## 🧠 Rationale behind the change

La decisión de diseño central es **qué se hace fuera del pipeline y qué dentro**:

| Tipo de operación | Ejemplos | Dónde | Por qué |
|---|---|---|---|
| Depende sólo de la fila | eliminar duplicados, invalidar valores fuera de dominio, descartar filas sin *target* | Antes del *split* | No aprende ningún parámetro de los datos |
| Aprende parámetros del conjunto | mediana de imputación, límites de atípicos, escala, bins, categorías | **Dentro del `Pipeline`**, ajustado sólo con `train` | Usar información de `test` inflaría artificialmente el desempeño |

Trade-offs considerados:

- **Atípicos: marcar y acotar en lugar de eliminar.** El dataset queda en 303 pacientes;
  los valores extremos de `chol`, `rest_bp` y `old_peak` son pacientes reales de alto
  riesgo. Se generan indicadores `<columna>_atipico` y se winsorizan los valores con los
  límites de Tukey aprendidos en `train`, en vez de borrar registros.
- **Selección de atributos conservadora.** La información mutua es univariada y ruidosa
  con ~240 filas, así que se promedian 15 repeticiones, se declaran los atributos
  discretos y se usa un umbral bajo. Sólo se descarta `fbs` (información mutua ≈ 0.001 y
  85 % de valor dominante).
- **`RobustScaler` sobre `StandardScaler`/`MinMaxScaler`**, coherente con conservar los
  pacientes extremos (se compara empíricamente en el notebook).
- **Deduplicación parcial.** Al anular los valores corruptos, un mismo paciente aparece
  varias veces con distinto patrón de celdas vacías y `drop_duplicates()` ya no lo
  detecta. Se eliminan las filas *subsumidas* por otra más completa: sin este paso,
  copias del mismo paciente caerían en `train` y en `test` a la vez.

## Type of changes

- [x] ✨ New Feature (changes that introduce new functionality)
- [x] ⚗️ Experiments (A notebook with experimentation results)

## 🛠 What does this PR implement

**`notebooks/4-feat_eng/04_feature_engineering.ipynb`** (64 celdas, ejecutado de principio
a fin sin errores).

**1. Limpieza de datos**

- Normalización de texto (espacios sobrantes, espaciado interno).
- Validación contra dominios declarados y rangos fisiológicos; todo valor corrupto
  (`'fggfds'`, `'2345'` en `sex`, …) pasa a faltante.
- Eliminación de filas sin *target* y de filas con más del 50 % de atributos ausentes.
- Eliminación de duplicados exactos **y parciales**: `3.030 → 303` registros únicos.
- Tabla de trazabilidad con las filas eliminadas en cada paso.
- Imputación (mediana / moda) **dentro del pipeline**, nunca antes del *split*.

**2. Selección de atributos**

- Información mutua promediada (15 repeticiones) + índice de cuasi-constancia, calculados
  sólo sobre `train`. Se descarta `fbs`.
- Verificación de duplicados emergentes tras reducir columnas.

**3. Ingeniería de atributos**

- Dos transformadores propios sobre `BaseEstimator` + `TransformerMixin`, con una clase
  base común que aprende los límites de Tukey: `RecortadorAtipicos` (winsorizing) y
  `MarcadorAtipicos` (indicadores binarios).
- 7 atributos derivados vía `FunctionTransformer`: `fc_maxima_teorica`,
  `reserva_cardiaca`, `chol_rel_edad`, `carga_presion`, `log_old_peak`, `sqrt_chol`,
  `st_deprimido`.
- Discretización en cuartiles de `age`, `chol` y `max_hr` con `KBinsDiscretizer`
  (`encode="onehot-dense"`), conservando además su versión continua.

**4. Escalado**

- `RobustScaler`, elegido tras comparar gráfica y numéricamente contra `StandardScaler` y
  `MinMaxScaler`.

**5. Encoding**

- `OneHotEncoder` (`handle_unknown="infrequent_if_exist"`) para `sex`, `chest_pain`,
  `rest_ecg` y `thal`.
- `OrdinalEncoder` con las categorías declaradas explícitamente para `slope` (1 < 2 < 3) y
  `ca` (0 ≤ … ≤ 3).

**Ensamblaje y validación**

- 6 sub-`Pipeline` unidos por un `ColumnTransformer`, envueltos en un `Pipeline` de dos
  pasos: `12 atributos → 42 atributos procesados`.
- Verificaciones automáticas (sin faltantes, sin infinitos, mismas columnas en `train` y
  `test`, sin varianza nula) que lanzan excepción si fallan.
- Prueba de no-fuga de datos comparando estadísticos de `train` vs `test`.
- Prueba de extremo a extremo con `LogisticRegression`: **ROC-AUC 0.886 ± 0.037** en
  validación cruzada estratificada de 5 *folds* sobre `train`.
- Persistencia del pipeline (`joblib`) y de los datasets procesados (`parquet`), con
  verificación de que el pipeline recargado reproduce exactamente la misma salida.

**Cambios de soporte**

- `notebooks/1-data/corazon.csv` + `datos_corazon_Info.txt`: dataset de entrada y su
  diccionario de datos.
- `.code_quality/ruff.toml`: reglas por archivo para notebooks (`E402`, `PLR2004`, `B905`,
  `RUF001`, `RUF005`), manteniendo activas las que sí importan.
- `pyproject.toml` / `uv.lock`: dependencias del notebook, incluida `pyarrow` para parquet.
- `.gitignore`: no versionar `models/*.joblib` (artefacto regenerable).

## 🙈 Missing

- Los transformadores personalizados están definidos en el notebook, así que el `.joblib`
  sólo se puede deserializar donde esas clases existan. Antes de desplegar deben moverse a
  `src/pipelines/feature_pipeline/` — queda documentado en el notebook y propuesto para la
  etapa 7.
- No se optimizan hiperparámetros del preprocesamiento (`n_bins`, `factor` del recorte,
  `strategy` de imputación): corresponde a la etapa 5 vía `GridSearchCV` sobre el pipeline
  completo.
- No se aplican técnicas de balanceo: el *target* está razonablemente balanceado
  (54 % / 46 %).

## 🧪 How should this be tested?

```bash
uv sync --all-extras --dev
uv run pytest --cov

# Ejecutar el notebook de principio a fin (debe correr sin errores):
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/4-feat_eng/04_feature_engineering.ipynb

# Calidad de código (igual que en el CI):
uvx pre-commit run --all-files
```

Puntos a revisar con especial atención:

1. Que ningún transformador se ajuste con datos de `test` (sección 4 y sección 13.2).
2. Que el criterio de descarte de atributos de la sección 5 les parezca razonable.
3. Que los atributos derivados de la sección 6.1 tengan sentido clínico.
