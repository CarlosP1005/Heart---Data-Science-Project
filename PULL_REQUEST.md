# Selección del mejor modelo con AutoML: FLAML + MLJar, validación honesta y modelo serializado

Closes #13

## ✨ Context

Sexta etapa del ciclo de vida del proyecto (`notebooks/5-models`), por la vía **automática**
que propone el issue: usar herramientas de AutoML en lugar de comparar modelos a mano.

El punto de partida son los dos números que dejó la etapa 5: el modelo base oficial es la
**heurística clínica con ROC-AUC 0.848 ± 0.055**, y la recomendación explícita de fijar el
criterio de éxito en **ROC-AUC > 0.90**.

Rama: `feature/seleccion-modelo-automl`, creada desde `main` siguiendo Gitflow.

**Resultado en una línea:** tres estrategias de búsqueda independientes convergen a una
**regresión logística con regularización L2 fuerte** (`C ≈ 0.033`) que alcanza **ROC-AUC
0.892 ± 0.039**. Supera al modelo base, **no de forma estadísticamente significativa**, y
**no llega al objetivo de 0.90**. El PR documenta por qué ése es el resultado honesto.

## 🧠 Rationale behind the change

**1. La decisión de diseño central: el AutoML propone, pero no juzga.**

Es la restricción que gobierna el notebook. Los motores de AutoML necesitan una matriz
numérica, así que hay que ajustar el pipeline de la etapa 4 sobre todo `train` antes de
entregársela. Eso introduce una **fuga leve**: las medianas de imputación y los límites de
Tukey que ve cada *fold* interno del buscador se calcularon con datos de los demás *folds*.

En lugar de ignorarlo o de esconderlo, el notebook lo declara (sección 6), aísla todas las
métricas del AutoML como *ordenamiento de candidatos*, y **vuelve a evaluar cada candidato
reconstruido como `Pipeline` de scikit-learn con el preprocesamiento dentro** (sección 12).
Después **cuantifica cuánto valía el optimismo** (sección 13).

No es un formalismo: **cambió el entregable**. FLAML declaró ganador a XGBoost con 0.9034 de
su validación interna; medido honestamente cae a 0.8901 y queda por detrás de la regresión
logística (0.8917). El optimismo resultó **proporcional a la flexibilidad del modelo** —+0.023
en Extra Trees, +0.013 en XGBoost, +0.005 en la logística—, así que no es un sesgo de nivel
inofensivo: **reordena el ranking**. Un uso ingenuo (`fit` → leer `best_estimator` → `dump`)
habría entregado otro modelo con un número más alto al lado, y ese número habría sido falso.

**2. La prueba estadística se corrige por solapamiento de *folds*, y eso cambia la conclusión.**

Los 30 *folds* de `RepeatedStratifiedKFold(5 × 6)` comparten pacientes, así que sus
diferencias no son observaciones independientes y una prueba t normal subestima la varianza.
Se usa la **corrección de Nadeau-Bengio**. La diferencia es de tres órdenes de magnitud:

| Comparación | p sin corregir (Wilcoxon) | p corregido |
|---|---|---|
| `automl_lrl2` vs modelo base | **0.0002** | **0.1502** |
| `automl_xgboost` vs modelo base | 0.0002 | 0.1295 |
| `automl_lrl2` vs `automl_xgboost` | — | 0.9203 |

Un notebook que reportara sólo el Wilcoxon concluiría "mejora altamente significativa". La
conclusión correcta es que **con 242 pacientes no hay evidencia suficiente** para afirmar que
el mejor modelo de AutoML supera a contar cuatro señales de riesgo en una hoja de papel.

**3. Cuando el desempeño empata, decide la parsimonia — y eso confirma la etapa 5.**

Los tres primeros candidatos son estadísticamente indistinguibles (p ≈ 0.92 y 0.83). Con el
criterio de desempeño agotado, se elige por simplicidad, sobreajuste y auditabilidad:

| | ROC-AUC (CV) | MCC | brecha train-val | hiperparámetros |
|---|---|---|---|---|
| **`automl_lrl2`** | **0.892 ± 0.039** | **0.656** | **0.031** | **1** |
| `automl_xgboost` | 0.890 ± 0.040 | 0.640 | 0.043 | 9 |
| `automl_ensamble` | 0.889 ± 0.039 | 0.614 | 0.043 | 13 |
| modelo base (etapa 5) | 0.848 ± 0.055 | 0.562 | 0.003 | 0 |

La recomendación 4 de la etapa 5 decía literalmente que si un *gradient boosting* no supera
claramente a la logística regularizada, el problema no lo necesita. **El AutoML no descubrió
un modelo mejor: convirtió esa conjetura en evidencia.**

**4. Se arregla el CI, que estaba roto en `main` antes de esta tarea.**

Al preparar el PR aparecieron dos problemas previos que impedían que los checks pasaran. No
son una sospecha: la última ejecución de *CI/CD Tests* sobre `main`
([run #37](https://github.com/CarlosP1005/Heart---Data-Science-Project/actions/runs/32450347904),
commit `e590d46`) está en **Failure**, con `test` en exit code 2 y `pre-commit` en exit code 1
—`actionlint` y el chequeo de *cruft* sí pasan—.

Van en commits separados para que se puedan revisar (o revertir) aparte del contenido de la
etapa:

- **`uv.lock` corrupto.** `uv sync` fallaba con `TOML parse error at line 1253` (la entrada de
  `pycparser` tenía un `wheels = [` duplicado de `pre-commit`), y además faltaban paquetes en
  el grafo. El job `test` no llegaba a ejecutar pytest. Se regeneró el lock y se declararon en
  `pyproject.toml` las dependencias que el código ya usaba sin declarar: `scikit-learn`,
  `scipy`, `joblib` y `seaborn`. También se corrigió `authors` al formato PEP 621
  (`RUF200`).
- **92 errores de `ruff` en notebooks ya en `main`** (01, 02, 03, 04 y 05). El job `pre-commit`
  fallaba. Se corrigieron los mecánicos (`B905` `zip(..., strict=False)`, `RUF005`) y el resto
  —`E402`, `PLR2004`, `RUF001`— se silenció con `# noqa` puntual, sin tocar la lógica.

## Type of changes

- [x] ✨ New Feature (changes that introduce new functionality)
- [x] ⚗️ Experiments (A notebook with experimentation results)
- [x] 🐛 Bug Fix (fixes an issue) — CI roto en `main`

## 🛠 What does this PR implement

### Archivos

| Archivo | Qué es |
|---|---|
| `notebooks/5-models/06_seleccion_modelo_automl-CJPS-2026-08-21.ipynb` | Notebook principal: 26 secciones, ejecutado de extremo a extremo con todas las salidas |
| `src/pipelines/training_pipeline/` | Configuración ganadora + `entrenar` / `predecir` / `guardar_artefacto` / `cargar_artefacto` |
| `tests/pipelines/training_pipeline/` | 7 pruebas del módulo anterior (umbral, ida y vuelta del `.joblib`, rechazo de artefactos ajenos) |
| `models/modelo_seleccionado_automl.joblib` | **Entregable del issue**: preprocesamiento + modelo + metadatos (19 KB) |
| `pyproject.toml`, `uv.lock` | Dependencias declaradas y lock regenerado |
| `.gitignore` | Excepción para versionar el `.joblib` entregable |
| Notebooks 01–05 | Sólo correcciones de lint para desbloquear el CI |

### Los ocho puntos del issue

| Punto | Cómo se resolvió | Sección |
|---|---|---|
| **Herramientas de AutoML** | **FLAML** (200 configuraciones, 6 familias, determinista) + **MLJar** (`Compete`, 26 modelos, *golden features*, apilado) + **TabPFN** documentado con su código y el motivo exacto por el que no se pudo ejecutar | 6, 7, 9, 10 |
| **Learning curve** | 8 tamaños × 15 evaluaciones por punto, con `return_times=True` | 15 |
| **Overfitting / underfitting** | Brecha train-val por modelo + *validation curves* de `C` y de `n_estimators` recorriendo el fenómeno completo | 16 |
| **Escalabilidad (tiempo y score)** | Dos niveles: presupuesto de búsqueda vs. score (5→120 s) y coste de entrenar/predecir vs. tamaño | 8 y 17 |
| **Almacenar pipeline + modelo** | Un solo `.joblib` con `Pipeline` completo y metadatos, verificando que recarga y reproduce sus predicciones | 21 |
| **Interpretar los resultados** | Coeficientes + importancia por permutación + SHAP, y su lectura conjunta | 20 y 22 |
| **Análisis de los resultados** | Escalera de desempeño, qué se puede y qué no se puede afirmar, validez de la comparación | 23 |
| **Recomendaciones** | 10 recomendaciones accionables | 25 |

### Hallazgos que conviene discutir en la revisión

1. **No se alcanzó el objetivo de 0.90 de la etapa 5.** El techo honesto es ~0.89. Se
   documenta como resultado, no como fracaso: sirve para recalibrar el objetivo con evidencia
   (recomendación 1) en lugar de con expectativas.

2. **El ganador declarado por el AutoML no es el ganador real** (sección 13). Es el hallazgo
   metodológico del PR y el argumento para adoptar este protocolo como estándar del proyecto.

3. **Ninguna mejora sobre el modelo base es estadísticamente significativa** (p > 0.10 con la
   corrección). La misma comparación sin corregir daría p < 0.001.

4. **El umbral importa más que el modelo.** Bajarlo de 0.50 a 0.36 —elegido maximizando F2
   **sólo en `train`**— lleva la sensibilidad en test de 0.929 a **1.000**, a costa de bajar la
   especificidad de 0.879 a 0.697. Ninguna búsqueda de hiperparámetros movió nada parecido. La
   elección del punto de operación es clínica, no estadística, y el notebook entrega la curva
   completa del intercambio.

5. **El modelo base gana en el conjunto de test** (ROC-AUC 0.9605 vs 0.9578). Con 61 pacientes
   eso es varianza, no señal — la misma varianza que hace que todos los modelos puntúen entre
   5 y 11 centésimas por encima de su validación cruzada. Se deja explícito porque es la mejor
   ilustración de por qué el modelo **no** se eligió mirando test.

6. **`ca` desplaza a `thal` como atributo más importante.** La etapa 5 eligió `thal` con dos
   criterios *univariados*; en el modelo multivariante manda `ca` (caída de ROC-AUC de 0.055
   al permutarlo, frente a 0.014 de `thal`). No es contradicción: es la diferencia entre
   importancia marginal y condicional. Coeficientes, permutación y SHAP coinciden, y los
   signos son clínicamente correctos.

7. **`chol` y `rest_ecg` tienen importancia por permutación negativa**: no aportan. Candidatos
   claros de simplificación para la siguiente etapa.

8. **El ensamble no aportó nada** (0.889 frente a 0.892 del mejor de sus componentes). Cuando
   los modelos se equivocan con los mismos pacientes, promediarlos no los corrige.

## 🙈 Missing

- **TabPFN no se ejecutó.** Sus pesos viven en un repositorio *gated* de HuggingFace y el
  entorno de ejecución no tiene ese acceso. Tampoco se añadió al grupo `automl` porque
  arrastra PyTorch (~1 GB) para un modelo que de todas formas no podría descargar sus pesos.
  El código queda listo en la sección 10 y la celda informa del motivo exacto en lugar de
  fallar en silencio.
- **No hay validación cruzada anidada.** Es la forma de eliminar —en lugar de sólo medir— la
  fuga de la sección 6, y queda como recomendación 4 para la siguiente etapa.
- **No se recalibraron las probabilidades.** El *Brier score* es aceptable (0.104) y la curva
  de calibración sigue la diagonal con desviaciones en los extremos. Si el proyecto llega a
  comunicar la probabilidad al paciente, habría que pasar por `CalibratedClassifierCV`.
- **`data/` sigue en `.gitignore`**, así que los CSV de `data/08_reporting/automl_*.csv` no se
  versionan: se regeneran ejecutando el notebook. El `.joblib` sí se versiona porque es un
  entregable explícito del issue (se añadió una excepción al `.gitignore`).

## 🧪 How should this be tested?

```bash
uv sync --all-extras --dev
uvx pre-commit run --all-files
uv run pytest --cov
```

Los tres deben pasar en limpio (14 pruebas, cobertura de `src/` al 98 %). **Antes de este PR,
`uv sync` fallaba y `pre-commit` reportaba 92 errores** — ver *Rationale* punto 4.

Para reproducir el notebook completo (15–25 minutos, casi todo en las dos búsquedas):

```bash
uv sync --all-extras --dev --group automl
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/5-models/06_seleccion_modelo_automl-CJPS-2026-08-21.ipynb
```

La búsqueda principal de FLAML está acotada por `max_iter=200` y **no** por reloj, así que es
determinista y debe reproducir las mismas cifras. El barrido de presupuestos de la sección 8 y
la búsqueda de MLJar sí dependen de la máquina.

El notebook **se autocomprueba**: la sección 21 lanza `ValueError` si el `.joblib` no reproduce
sus predicciones al recargarse, y la 21.1 falla si `src/pipelines/training_pipeline/` se
desincroniza de la búsqueda. Una ejecución sin excepciones es en sí misma la prueba.

## 👀 Qué mirar primero en la revisión

Si el tiempo es corto, las tres secciones que sostienen todo lo demás son:

- **Sección 12** — la tabla de validación cruzada honesta, que es la que decide.
- **Sección 13** — por qué el ganador declarado por el AutoML no es el ganador real.
- **Sección 14** — por qué ninguna mejora resiste una prueba estadística correcta.
