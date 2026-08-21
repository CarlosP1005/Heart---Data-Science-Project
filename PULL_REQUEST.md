# Exploración inicial de datos: esquema, unificación de nulos y persistencia en parquet

Closes #9

## ✨ Context

Segunda etapa del ciclo de vida del proyecto (`notebooks/2-exploration`), siguiendo la guía
del curso [Ciencia de Datos en Producción — Exploración](https://joserzapata.github.io/post/ciencia-datos-proyecto-python/2-exploration/).

La carpeta `notebooks/2-exploration/` estaba vacía. Este PR cubre el hueco: antes de
analizar distribuciones o entrenar nada hay que responder una pregunta más básica —**¿qué
hay realmente en este archivo y de qué tipo es cada cosa?**—. La respuesta corta es que una
lectura normal con `pd.read_csv` devuelve **13 de las 14 columnas como texto**, así que
literalmente ninguna operación numérica funciona sobre los datos crudos.

Rama: `feature/exploracion-inicial`, creada desde `main` siguiendo Gitflow.

## 🧠 Rationale behind the change

**1. La decisión de diseño central: esta etapa no elimina ni una sola fila.**

Es la restricción que gobierna todo el notebook y conviene defenderla explícitamente,
porque es tentador aprovechar el viaje para limpiar de una vez. El archivo de salida tiene
exactamente las mismas 3 030 filas que el de entrada; lo único que cambia es **cómo están
representados** los datos.

El motivo es que eliminar duplicados, imputar faltantes o descartar registros incompletos
son decisiones que **dependen del modelo que se vaya a entrenar**, y por lo tanto pertenecen
a la etapa 4. Adelantarlas aquí haría irreversible una transformación que debería ser sólo
de formato. Con esta frontera, la salida de la etapa 2 es auditable: se puede demostrar que
no se perdió información, y de hecho el notebook lo verifica.

**2. Se lee todo con `dtype=str` y `keep_default_na=False`.**

Es lo contrario de lo que hace `read_csv` por defecto, y es deliberado. Si pandas infiere
los tipos, una sola celda corrupta convierte silenciosamente una columna numérica en texto y
el problema queda invisible. Leyendo todo literalmente, la inferencia de tipos pasa a ser una
**decisión explícita y documentada** del notebook en lugar de un efecto secundario. Y sin
`keep_default_na=False` no se podría ni siquiera ver qué convenciones de nulo usa el archivo,
que es el punto 2 del requerimiento.

**3. Tipos *nullable* de pandas en lugar de `float64` para todo.**

La conversión ingenua deja todo en coma flotante, lo que miente sobre la naturaleza del
dato (`ca = 2.0` vasos sanguíneos), pierde la intención de las booleanas y desperdicia
memoria. Se usan `Int16`, `Int8`, `Float32`, `boolean` y `category`, que admiten `pd.NA` sin
degradar a *float*. Es la única forma de tener una columna **entera y con faltantes a la
vez**.

**4. Parquet, con la contraprueba incluida.**

El notebook no se limita a afirmar que parquet es "un formato adecuado": guarda el mismo
DataFrame en CSV y compara. El resultado es que **0 de 14 tipos sobreviven al CSV y 14 de 14
sobreviven al parquet**. Sin esa evidencia, la elección de formato parecería administrativa;
con ella queda claro que es lo que hace reutilizable todo el trabajo de la sección 5.

## Type of changes

- [x] ✨ New Feature (changes that introduce new functionality)
- [x] ⚗️ Experiments (A notebook with experimentation results)

## 🛠 What does this PR implement

### Archivos

| Archivo | Qué es |
|---|---|
| `notebooks/2-exploration/02_exploracion_inicial-CJPS-2026-08-21.ipynb` | Notebook principal: 10 secciones, ejecutado de extremo a extremo con todas las salidas |
| `PULL_REQUEST.md` | Esta descripción |

No se toca ningún archivo existente de código ni de configuración.

### Los cuatro puntos del issue

| Punto | Cómo se resolvió | Resultado |
|---|---|---|
| **Descripción general** | Lectura como texto, comparación entre tipos inferidos y esperados, resumen por columna, análisis de duplicados | 3 030 × 14; con lectura normal 13 de 14 columnas vuelven como texto |
| **Unificar nulos** | Búsqueda exhaustiva de 17 tokens de nulo + invalidación contra dominios y rangos | Una sola representación (`NaN`): 1 647 visibles + **33 ocultos** |
| **Tipos correctos** | `Int16`, `Int8`, `Float32`, `boolean`, `category` | 0 columnas de texto; memoria **−82 %** |
| **Formato adecuado** | `.parquet` + *snappy*, con verificación de ida y vuelta y contraprueba contra CSV | **−91 %** de tamaño; esquema intacto al releer |

### Hallazgos que conviene discutir en la revisión

1. **El 81 % del archivo son duplicados exactos.** Las 3 030 filas contienen sólo **568**
   combinaciones distintas, más 15 filas completamente vacías. El tamaño real del conjunto
   es mucho menor de lo que aparenta. Se documenta aquí y se resuelve en la etapa 4.

2. **Hay 33 faltantes disfrazados de dato válido.** Cadenas como `'fggfds'` o números como
   `2345` en columnas de texto. `df.isna()` los reporta como presentes; sólo se detectan
   validando contra el diccionario de datos. **Cinco de ellos están en la variable
   objetivo** (`'fsg'`, `'gsfdg'`, `'g'`, `'sf'`, `'fsdg'`), lo que los vuelve registros
   inservibles para entrenar y para evaluar.

3. **Un espacio invisible podía destruir el 46 % de una columna.** Las **1 387** apariciones
   de `left ventricular hypertrophy ` llevan un espacio final — es decir, *todas* las de esa
   categoría. Validar contra el dominio sin normalizar antes las habría convertido en
   faltantes, sin ningún mensaje de error. De ahí que el notebook insista en el orden:
   normalizar primero, validar después.

4. **Los rangos fisiológicos no descartaron nada.** Toda la basura era texto sin sentido, no
   dígitos mal tecleados: los valores convertibles ya estaban dentro de lo plausible (edad
   29-77, presión 94-200, colesterol 126-564). La validación por rango se conserva como red
   de seguridad, pero es honesto registrar que en este archivo no fue la que atrapó nada.

5. **Verificación cruzada con `src/pipelines/feature_pipeline/`.** El notebook declara sus
   dominios a partir del diccionario de datos, y una celda comprueba que **coinciden con los
   del módulo** que usan las etapas 4 y 5, fallando si divergen. La única diferencia
   detectada es benigna y va en la dirección segura: el módulo acepta `atypical` en
   `chest_pain`, una categoría que no aparece ni una vez en el archivo.

## 🙈 Missing

- No se eliminan duplicados ni filas vacías, no se imputa nada y no se descartan los
  registros sin *target*: es deliberado, corresponde a la etapa 4 (ver *Rationale* punto 1).
- No hay análisis de distribuciones, correlaciones ni relación con el *target*: eso es la
  etapa 3.
- `data/` está en `.gitignore`, así que `corazon_tipificado.parquet` **no se versiona**. Se
  regenera ejecutando el notebook.

## 🧪 How should this be tested?

```bash
uv sync --all-extras --dev
uvx pre-commit run --all-files
```

Para reproducir el notebook completo (~15 segundos):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/2-exploration/02_exploracion_inicial-CJPS-2026-08-21.ipynb
```

Debe generar `data/02_intermediate/corazon_tipificado.parquet` (≈19 KB) y todas las
verificaciones de las secciones 6, 7.1 y 8 deben salir en `OK`. El notebook **lanza
`ValueError` si alguna falla**, así que una ejecución sin excepciones es en sí misma la
prueba.
