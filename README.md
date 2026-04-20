# Agente Doom

Proyecto de aprendizaje por refuerzo para entrenar y evaluar un agente en escenarios de ViZDoom usando Gymnasium, Stable-Baselines3 y `RecurrentPPO` con LSTM.

## Requisitos

- Windows x64.
- Python 3.10 o superior. El proyecto fue probado localmente con Python 3.14.2.
- El lanzador `py` disponible en PowerShell.

No es necesario activar el entorno virtual con `Activate.ps1`. En Windows puede fallar por la politica de ejecucion de scripts; por eso los comandos de este README usan directamente el ejecutable del entorno virtual.

## Instalacion

Desde PowerShell:

```powershell
cd E:\agente-doom
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ejecutar entrenamiento

Entrenamiento completo:

```powershell
cd E:\agente-doom
.\.venv\Scripts\python.exe src\train.py
```

Entrenamiento rapido para pruebas:

```powershell
cd E:\agente-doom
.\.venv\Scripts\python.exe src\train.py --config fast
```

Entrenamiento eficiente, recomendado para aprender mejor sin gastar tiempo en ventana ni video:

```powershell
cd E:\agente-doom
.\.venv\Scripts\python.exe src\train.py --config efficient
```

El entrenamiento usa las configuraciones definidas en `src\config.py`.

Valores importantes:

- `env_name`: escenario de ViZDoom a usar.
- `total_timesteps`: cantidad total de pasos de entrenamiento.
- `render`: muestra o no la ventana de ViZDoom.
- `record`: guarda o no video durante entrenamiento.

Por defecto se entrena con:

```python
"env_name": "basic.cfg"
"total_timesteps": 500000
"render": True
"record": True
```

La configuracion rapida usa:

```python
"n_steps": 256
"n_epochs": 3
"total_timesteps": 10000
"render": False
"record": False
```

La configuracion eficiente usa:

```python
"n_steps": 1024
"n_epochs": 5
"total_timesteps": 100000
"render": False
"record": False
```

Tambien puedes sobrescribir los timesteps solo para una ejecucion:

```powershell
.\.venv\Scripts\python.exe src\train.py --config fast --timesteps 256
```

Por ejemplo, para probar la configuracion eficiente sin esperar el entrenamiento completo:

```powershell
.\.venv\Scripts\python.exe src\train.py --config efficient --timesteps 1024
```

Al terminar, el modelo se guarda en:

```text
logs\checkpoints\ppo_doom_recurrent.zip
```

El entrenamiento rapido se guarda en:

```text
logs\checkpoints\ppo_doom_recurrent_fast.zip
```

El entrenamiento eficiente se guarda en:

```text
logs\checkpoints\ppo_doom_recurrent_efficient.zip
```

Tambien se generan logs de TensorBoard en `logs\` y videos en `data\videos\`.

## Guardado durante entrenamiento

`train.py` guarda el modelo al terminar normalmente y tambien intenta guardarlo si interrumpes con `Ctrl+C` o si ViZDoom se cierra.

El ultimo progreso de cada perfil se guarda en:

```text
logs\checkpoints\ppo_doom_recurrent.zip
logs\checkpoints\ppo_doom_recurrent_fast.zip
logs\checkpoints\ppo_doom_recurrent_efficient.zip
```

Ademas, se crean checkpoints automaticos en:

```text
logs\checkpoints\auto\
```

Frecuencia de guardado automatico:

```text
default:   cada 50000 pasos
fast:      cada 2000 pasos
efficient: cada 10000 pasos
```

Si cierras la terminal de golpe, apagas la PC o matas el proceso, Python puede no alcanzar a ejecutar el guardado final. En ese caso usa el checkpoint automatico mas reciente de `logs\checkpoints\auto\`.

## Ejecutar evaluacion

Primero debe existir un modelo entrenado en:

```text
logs\checkpoints\ppo_doom_recurrent.zip
```

Luego ejecuta:

```powershell
cd E:\agente-doom
.\.venv\Scripts\python.exe src\evaluate.py
```

Para evaluar el checkpoint del entrenamiento rapido:

```powershell
cd E:\agente-doom
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint ppo_doom_recurrent_fast
```

Para evaluar el checkpoint del entrenamiento eficiente:

```powershell
cd E:\agente-doom
.\.venv\Scripts\python.exe src\evaluate.py --checkpoint ppo_doom_recurrent_efficient
```

La evaluacion carga el checkpoint y ejecuta el agente con `render=True`. Para salir, presiona `Ctrl+C`.

## Ver TensorBoard

```powershell
cd E:\agente-doom
.\.venv\Scripts\tensorboard.exe --logdir logs
```

Luego abre en el navegador la URL que muestre TensorBoard, normalmente:

```text
http://localhost:6006
```

## Escenarios disponibles

Los escenarios estan en `data\scenarios\`:

```text
basic.cfg
deadly_corridor.cfg
defend_the_center.cfg
health_gathering.cfg
```

Para cambiar de escenario, edita `env_name` en `src\config.py`:

```python
"env_name": "defend_the_center.cfg"
```

## Prueba rapida de entorno

Este comando valida que ViZDoom puede iniciar, resetear y ejecutar un paso:

```powershell
cd E:\agente-doom
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from config import CONFIG, SCENARIOS_DIR, VIDEO_DIR, LOG_DIR; from environment import make_doom_env; cfg={**CONFIG, 'SCENARIOS_DIR': SCENARIOS_DIR, 'VIDEO_DIR': VIDEO_DIR, 'LOG_DIR': LOG_DIR, 'render': False}; env=make_doom_env(cfg); obs=env.reset(); print('reset ok', obs.shape); obs, rewards, dones, infos = env.step([0]); print('step ok', obs.shape, rewards, dones); env.close()"
```

## Problemas comunes

### `python` no se reconoce como comando

Usa `py` para crear el entorno:

```powershell
py -m venv .venv
```

Despues usa siempre:

```powershell
.\.venv\Scripts\python.exe
```

### PowerShell bloquea `Activate.ps1`

No necesitas activar el entorno. Ejecuta los comandos con la ruta completa:

```powershell
.\.venv\Scripts\python.exe src\train.py
```

### Falta `moviepy`

`moviepy` es necesario porque el entrenamiento usa `VecVideoRecorder` para guardar videos. Ya esta incluido en `requirements.txt`.

### Fallo instalando `ale-py`

Este proyecto no necesita Atari/ALE. Por eso `requirements.txt` usa `stable-baselines3` sin el extra `[extra]`, evitando compilar `ale-py` en Windows.

## Estructura principal

```text
src\config.py        Configuracion general
src\environment.py   Wrapper Gymnasium para ViZDoom
src\model.py         Modelo RecurrentPPO con extractor CNN
src\train.py         Entrenamiento
src\evaluate.py      Evaluacion del checkpoint
data\scenarios\      Escenarios .cfg y .wad
logs\                Logs y checkpoints generados
data\videos\         Videos generados durante entrenamiento
```
