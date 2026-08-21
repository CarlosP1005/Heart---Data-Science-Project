# Modelo base (*baseline*): dummies, regla de una variable y heurística clínica

## ✨ Context

Quinta etapa del ciclo de vida del proyecto (`notebooks/5-models`), siguiendo la guía del
curso [Ciencia de Datos en Producción — Baseline Model](https://joserzapata.github.io/post/ciencia-datos-proyecto-python/5-baseline_model/).

Con el pipeline de preparación ya construido en la etapa 4 (PR #18), falta lo más
importante antes de entrenar cualquier modelo: **saber contra qué se va a comparar**. Sin
un punto de referencia, un 80 % de exactitud parece un buen resultado hasta que uno
descubre que responder siempre "sano" ya acierta el 54 %. Este PR entrega el notebook que
fija ese punto de referencia y, de paso, verifica que la tubería completa (datos →
preprocesamiento → estimador → evaluación) funciona de extremo a extremo.

Rama: `5-baseline`, creada desde `main` siguiendo Gitflow.

## 🧠 Rationale behind the change

**1. Se eligió una heurística clínica como modelo base oficial, no un `DummyClassifier`.**

El requerimiento admite ambas opciones. El *dummy* fija el listón en ROC-AUC 0.500, que es
un adversario trivial: cualquier modelo lo supera y eso no informa nada. La heurística
—contar cuatro señales de riesgo (`chest_pain == asymptomatic`, `exang == 1`,
`old_peak > 1.0`, `ca > 0`) sin aprender ningún parámetro— lo fija en **0.848**. Un modelo
de *machine learning* que no supere ese número no justifica su costo de mantenimiento
frente a una lista de chequeo en papel. Los cinco *dummy* se conservan documentados como
piso absoluto y como prueba de sanidad del montaje experimental.

**2. El pipeline de la etapa 4 se movió del notebook a `src/pipelines/feature_pipeline/`.**

Este es el cambio estructural del PR y merece justificación, porque toca código ya
mergeado. En la etapa 4 los transformadores personalizados vivían dentro del notebook, lo
que tenía dos consecuencias:

- El artefacto serializado `models/pipeline_feature_engineering.joblib` **no se puede
  deserializar** sin volver a ejecutar las celdas que definen `RecortadorAtipicos` y
  `MarcadorAtipicos`, porque pickle los referencia como `__main__.<clase>`.
- La alternativa —copiar y pegar el código a este notebook— habría creado dos versiones
  del mismo pipeline que se desincronizan a la primera corrección.

El código es **el mismo**; lo que cambia es dónde vive. Se verifica en el notebook que
`limpiar_datos()` reproduce el dataset limpio de la etapa 4 de forma idéntica (303 × 14).
Ahora el pipeline es importable, versionado y cubierto por pruebas en cada *push*.

**3. Validación cruzada repetida (5 × 6) en lugar de 5 *folds* simples.**

Con 242 pacientes de entrenamiento, cada *fold* de validación tiene ~48 casos. Una sola
pasada de 5 particiones produce desviaciones estándar tan ruidosas que dos modelos
separados por 3 puntos parecerían indistinguibles. Se usan 30 evaluaciones por modelo.

**4. ROC-AUC como métrica de decisión, con 10 métricas más reportadas.**

El problema tiene costo asimétrico (un falso negativo es un paciente enfermo enviado a
casa). Se reporta un panel de 11 métricas porque **ninguna sirve sola**, y el notebook lo
demuestra con un contraejemplo: `dummy_todos_enfermos` alcanza sensibilidad 1.000 y F1
0.629 con MCC 0.000.

## Type of changes

- [x] ✨ New Feature (changes that introduce new functionality)
- [x] ⚗️ Experiments (A notebook with experimentation results)
- [x] ✅ Tests (Unit tests, integration tests, end-to-end tests)

## 🛠 What does this PR implement

### Archivos

| Archivo | Qué es |
|---|---|
| `notebooks/5-models/05_baseline_model-CJPS-2026-08-21.ipynb` | Notebook principal: 17 secciones, ejecutado de extremo a extremo con todas las salidas |
| `src/pipelines/feature_pipeline/feature_pipeline.py` | Pipeline de la etapa 4 como módulo importable: contratos de datos, limpieza, transformadores y fábrica |
| `src/pipelines/feature_pipeline/__init__.py` | Superficie pública del módulo |
| `tests/pipelines/feature_pipeline/test_feature_pipeline.py` | 7 pruebas del módulo (98 % de cobertura de líneas) |

### Contenido del notebook

- **Reutilización del pipeline de la etapa anterior**, verificando que reproduce el
  dataset limpio de forma idéntica.
- **Partición train/test** 80/20 estratificada, con la misma semilla de la etapa 4 (242 /
  61 pacientes) para que los resultados sean comparables entre etapas.
- **11 métricas** con justificación clínica de cada una: exactitud, exactitud balanceada,
  precisión, sensibilidad, especificidad, F1, F2, ROC-AUC, PR-AUC, MCC y *Brier score*.
- **Validación cruzada** `RepeatedStratifiedKFold(5 × 6)` con media y desviación estándar
  reportadas para cada métrica y cada modelo.
- **Selección de la variable más importante** triangulando dos criterios independientes
  (información mutua y ROC-AUC de un modelo univariado en validación cruzada).
- **Curvas de aprendizaje** con `learning_curve(..., return_times=True)`.
- **Gráficas de escalabilidad**: tiempo de ajuste vs. n, tiempo de evaluación vs. n y
  score vs. tiempo de ajuste.
- **Interpretación, análisis, conclusiones, recomendaciones y propuestas** (secciones 12
  a 17).

### Resultados principales (validación cruzada, 30 evaluaciones)

| Modelo | Exactitud | Sensibilidad | ROC-AUC | MCC |
|---|---|---|---|---|
| `dummy_mayoritaria` | 0.541 ± 0.006 | 0.000 | **0.500 ± 0.000** | 0.000 |
| `dummy_todos_enfermos` | 0.459 ± 0.006 | **1.000** | 0.500 ± 0.000 | 0.000 |
| `regla_1var(thal)` | 0.768 ± 0.060 | 0.719 ± 0.089 | 0.765 ± 0.061 | 0.536 |
| **`heuristica_clinica`** (modelo base) | 0.781 ± 0.061 | 0.775 ± 0.090 | **0.848 ± 0.055** | 0.562 |
| `referencia_reg_logistica` | 0.804 ± 0.062 | 0.743 ± 0.086 | 0.883 ± 0.045 | 0.609 |

**Variable más importante: `thal`** (ROC-AUC univariado 0.765 ± 0.061, información mutua
0.156 nats). Primera por ambos criterios, y con justificación clínica: mide directamente
la perfusión miocárdica en lugar de ser un factor de riesgo indirecto.

### Hallazgos que conviene discutir en la revisión

1. **El margen para el *machine learning* es estrecho.** Una regla escrita a mano llega a
   0.848 y una regresión logística sobre los 42 atributos procesados a 0.883: 3.5 puntos,
   con bandas de una desviación estándar que se solapan.
2. **La curva de aprendizaje se aplana casi de inmediato.** La validación de la regresión
   logística pasa de 0.867 con 19 pacientes a 0.876 con 193. Multiplicar por diez los
   datos aportó menos de un punto. El techo lo pone la información de las variables, no el
   tamaño de la muestra — conviene invertir en mejores variables antes que en más filas.
3. **`thal` podría tener circularidad diagnóstica.** Si la gammagrafía formó parte del
   proceso que generó la etiqueta `disease`, su poder predictivo está inflado. Queda
   registrado como hipótesis a verificar, no como hecho.

## 🙈 Missing

- No se optimiza ningún hiperparámetro: corresponde a la etapa 5.1 (entrenamiento).
- La regresión logística aparece sólo como cota superior de referencia, no como modelo
  candidato; no se ajusta ni se regulariza.
- El umbral de decisión se deja en 0.5 por defecto. El notebook recomienda ajustarlo con
  criterio clínico, pero no lo hace aquí.
- `models/modelo_base.joblib` y los CSV de `data/08_reporting/` se generan al ejecutar el
  notebook y no se versionan (están en `.gitignore`).

## 🧪 How should this be tested?

```bash
uv sync --all-extras --dev
uv run pytest --cov          # 7 pruebas nuevas, 98 % de cobertura del módulo
uvx pre-commit run --all-files
```

Para reproducir el notebook completo (~1 minuto):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/5-models/05_baseline_model-CJPS-2026-08-21.ipynb
```

Todo es determinista con `random_state=42`; los números deben coincidir exactamente con
los del notebook versionado.
