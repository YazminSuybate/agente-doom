import argparse
import os

from stable_baselines3.common.callbacks import CheckpointCallback
from vizdoom import ViZDoomUnexpectedExitException

from config import LOG_DIR, SCENARIOS_DIR, VIDEO_DIR, TRAINING_CONFIGS
from environment import make_doom_env
from model import get_recurrent_model


def build_config(config_name, total_timesteps=None):
    config = dict(TRAINING_CONFIGS[config_name])
    if total_timesteps is not None:
        config["total_timesteps"] = total_timesteps

    return {
        **config,
        "SCENARIOS_DIR": SCENARIOS_DIR,
        "VIDEO_DIR": VIDEO_DIR,
        "LOG_DIR": LOG_DIR,
    }


def train(config_name="default", total_timesteps=None):
    for path in [LOG_DIR, VIDEO_DIR]:
        os.makedirs(path, exist_ok=True)

    config = build_config(config_name, total_timesteps)
    save_path = os.path.join(LOG_DIR, "checkpoints", config["save_name"])
    auto_save_dir = os.path.join(LOG_DIR, "checkpoints", "auto")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(auto_save_dir, exist_ok=True)

    env = make_doom_env(config, record=config["record"])
    model = get_recurrent_model(env, config)
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=auto_save_dir,
        name_prefix=config["save_name"],
    )

    print(
        f"Iniciando entrenamiento '{config_name}' en {config['env_name']} "
        f"por {config['total_timesteps']} timesteps..."
    )
    print(
        "Checkpoints automaticos cada "
        f"{config['checkpoint_freq']} pasos en {auto_save_dir}"
    )

    completed = False
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            tb_log_name=config["tb_log_name"],
            callback=checkpoint_callback,
        )
        completed = True
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido por el usuario. Guardando progreso...")
    except ViZDoomUnexpectedExitException:
        print("Entrenamiento detenido: ViZDoom se cerro. Guardando progreso...")
    finally:
        model.save(save_path)
        try:
            env.close()
        except ViZDoomUnexpectedExitException:
            pass

        if completed:
            print(f"Entrenamiento completado. Modelo guardado en {save_path}")
        else:
            print(f"Progreso guardado en {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el agente ViZDoom.")
    parser.add_argument(
        "--config",
        choices=sorted(TRAINING_CONFIGS.keys()),
        default="default",
        help="Configuracion de entrenamiento a usar.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Sobrescribe total_timesteps para esta ejecucion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_name=args.config, total_timesteps=args.timesteps)
