# Agente Doom

Proyecto de aprendizaje por refuerzo para entrenar y evaluar un agente en escenarios de ViZDoom con `Gymnasium`, `Stable-Baselines3` y `RecurrentPPO`.

## Cambios principales

- La logica ahora esta separada por responsabilidades en `config`, `envs`, `models`, `services`, `utils` y `shared`.
- Los perfiles y escenarios viven fuera del codigo en `configs/training_profiles.toml`.
- El espacio de acciones ahora soporta combinaciones de botones y `no-op` mediante `MultiDiscrete`, en vez de limitar al agente a una sola tecla por paso.
- El reward shaping es configurable por escenario sin modificar el wrapper del entorno.
- Cada perfil puede fijar una `seed` reproducible y tambien sobrescribirse por CLI.
- Cada checkpoint nuevo guarda un archivo `.json` con la configuracion usada para entrenarlo.
- El entrenamiento reanuda automaticamente desde el ultimo checkpoint compatible, salvo que uses `--from-scratch`.
- Cada corrida puede evaluar periodicamente y guardar `best_model` aparte del ultimo estado.
- El entrenamiento ahora soporta `early stopping` por evaluaciones sin mejora.
- El proyecto soporta perfiles de `curriculum training` con multiples etapas por escenario.
- Existe un runner secuencial de sweeps para comparar combinaciones de hiperparametros.
- Cada entrenamiento genera un reporte JSON en `artifacts/reports/` y mantiene un indice de corridas.
- Existe una CLI unificada con subcomandos para entrenar, evaluar, listar perfiles, listar checkpoints e inspeccionar metadata.
- Los artefactos generados ya no viven mezclados con el codigo: ahora se escriben en `artifacts/`.
- El tooling del proyecto vive en `pyproject.toml` y se puede automatizar con `pre-commit`.
- Se agregaron pruebas automatizadas para perfiles, metadata de checkpoints, reportes y smoke test del entorno.

## Requisitos

- Windows x64.
- Python 3.12.
- `py` disponible en PowerShell.

## Instalacion

```powershell
cd E:\agente-doom
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Estructura

```text
src\
  doom_agent\
    cli\         Entradas de linea de comandos
    config\      Carga tipada del catalogo TOML y rutas del proyecto
    envs\        Wrapper Gymnasium, acciones y reward shaping
    models\      Extractor CNN y fabrica del modelo PPO recurrente
    services\    Casos de uso de entrenamiento y evaluacion
    shared\      Tipos reutilizables
    utils\       Checkpoints, archivos y metadata
  train.py       Punto de entrada compatible hacia atras
  evaluate.py    Punto de entrada compatible hacia atras
configs\         Catalogo TOML de perfiles y escenarios
tests\           Pruebas automatizadas
data\scenarios\  Escenarios .cfg y .wad
artifacts\       Checkpoints, TensorBoard, reportes y videos generados
```

## CLI Unificada

Listar perfiles y escenarios:

```powershell
.\.venv\Scripts\python.exe src\cli.py list-profiles
```

Listar checkpoints conocidos:

```powershell
.\.venv\Scripts\python.exe src\cli.py list-checkpoints --limit 10
```

Listar corridas registradas:

```powershell
.\.venv\Scripts\python.exe src\cli.py list-runs --limit 10
```

Inspeccionar metadata de un checkpoint:

```powershell
.\.venv\Scripts\python.exe src\cli.py inspect-checkpoint --checkpoint ppo_doom_recurrent_fast --select best
```

Ejecutar un sweep secuencial:

```powershell
.\.venv\Scripts\python.exe src\cli.py sweep --config fast --learning-rates 0.0001,0.0003 --n-steps-values 256,512 --batch-sizes 64 --seeds 42,43
```

## Entrenamiento

Perfil por defecto:

```powershell
.\.venv\Scripts\python.exe src\train.py
```

Perfil rapido:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast
```

Perfil rapido en otro escenario:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast --scenario deadly_corridor
```

Entrenar un perfil con curriculum:

```powershell
.\.venv\Scripts\python.exe src\train.py --config curriculum_fast
```

Sobrescribir la `seed` del perfil:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast --seed 123
```

Forzar una corrida desde cero:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast --from-scratch
```

Reanudar desde un checkpoint explicito:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast --resume artifacts\checkpoints\ppo_doom_recurrent_fast.zip
```

Activar evaluacion periodica mas frecuente:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast --eval-freq 256 --eval-episodes 3
```

Activar o aprovechar early stopping desde el perfil:

```powershell
.\.venv\Scripts\python.exe src\train.py --config curriculum_fast --from-scratch
```

Perfil eficiente:

```powershell
.\.venv\Scripts\python.exe src\train.py --config efficient
```

Sobrescribir timesteps solicitados:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast --timesteps 256
```

Notas importantes:

- `RecurrentPPO` entrena por bloques de `n_steps`, asi que el total efectivo puede ser mayor al solicitado.
- El comando imprime ambos valores: `requested_timesteps` y `effective_timesteps`.
- Los checkpoints finales se guardan en `artifacts\checkpoints\`.
- Los checkpoints automaticos se guardan en `artifacts\checkpoints\auto\`.
- El ultimo estado entrenado se guarda como `<checkpoint>.zip`.
- El mejor modelo segun evaluacion periodica se guarda como `<checkpoint>_best.zip`.
- Cada checkpoint nuevo genera un archivo `.json` con la metadata del entrenamiento.
- Cada corrida genera un reporte en `artifacts\reports\`.
- El indice acumulado de corridas vive en `artifacts\reports\index.json`.
- Si usas `--scenario` distinto de `basic`, el nombre del checkpoint incluye un sufijo como `__deadly_corridor` para evitar colisiones.
- Por defecto, una nueva ejecucion intenta continuar desde el ultimo checkpoint compatible del mismo perfil y escenario.
- Si el perfil tiene `curriculum`, cada etapa se entrena secuencialmente y la siguiente reanuda desde el checkpoint final de la etapa previa.

## Evaluacion

Evaluar el checkpoint principal:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py
```

Por defecto, si existe, la evaluacion intentara usar `<checkpoint>_best.zip`.

Evaluar otro checkpoint:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint ppo_doom_recurrent_fast --steps 500
```

Forzar el ultimo estado entrenado en lugar del mejor:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --config fast --select last
```

Tambien puedes pasar una ruta directa:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint artifacts\checkpoints\ppo_doom_recurrent_fast.zip
```

Evaluar un checkpoint pero forzando otro escenario:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint ppo_doom_recurrent_fast --scenario defend_the_center
```

La evaluacion intenta usar la metadata del checkpoint para reconstruir el escenario y la configuracion correcta. Si el checkpoint es legado y no tiene metadata, usa el escenario del perfil `default` y trata de inferir el tipo de espacio de acciones.

## TensorBoard

```powershell
.\.venv\Scripts\tensorboard.exe --logdir artifacts\tensorboard
```

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Tipado

```powershell
.\.venv\Scripts\python.exe -m mypy
```

El proyecto usa `mypy` con reglas estrictas sobre `doom_agent`, `tests` y los wrappers legacy. La configuracion vive en `pyproject.toml`.

## Lint

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

## Pre-commit

```powershell
.\.venv\Scripts\python.exe -m pre_commit install
```

## Escenarios disponibles

```text
basic.cfg
deadly_corridor.cfg
defend_the_center.cfg
health_gathering.cfg
```

Para cambiar perfiles, escenarios o reward shaping sin tocar Python, edita `configs\training_profiles.toml`.
Tambien puedes definir ahi configuraciones de `early_stopping` y listas de `curriculum`.

## Artefactos generados

El proyecto ignora por Git:

- `artifacts/`
- `logs/`
- `data/videos/`
- `_vizdoom.ini`

Eso evita versionar videos, checkpoints, eventos de TensorBoard y archivos generados por ViZDoom.

## CI

Hay un workflow en `.github/workflows/ci.yml` con tres jobs separados en Windows:

- `lint`: instala dependencias y ejecuta `ruff`.
- `typing`: instala dependencias y ejecuta `mypy`.
- `tests`: instala dependencias y ejecuta la suite de pruebas.
