import os
import argparse

from sb3_contrib import RecurrentPPO
from vizdoom import ViZDoomUnexpectedExitException

from config import CONFIG, LOG_DIR, SCENARIOS_DIR
from environment import make_doom_env


def evaluate(checkpoint="ppo_doom_recurrent", steps=None):
    config_eval = {
        **CONFIG,
        "SCENARIOS_DIR": SCENARIOS_DIR,
        "render": True,
    }

    env = make_doom_env(config_eval)

    model_path = os.path.join(LOG_DIR, "checkpoints", checkpoint)
    if not os.path.exists(model_path + ".zip"):
        print("Error: no existe un modelo entrenado en la ruta especificada.")
        env.close()
        return

    model = RecurrentPPO.load(model_path, env=env)

    obs = env.reset()
    lstm_states = None
    episode_starts = [True]

    print("Evaluando agente... Presiona Ctrl+C para salir.")
    try:
        step_count = 0
        while steps is None or step_count < steps:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, rewards, dones, info = env.step(action)
            episode_starts = dones
            env.render()
            step_count += 1
        print(f"Evaluacion completada: {step_count} pasos ejecutados.")
    except KeyboardInterrupt:
        print("Evaluacion detenida por el usuario.")
    except ViZDoomUnexpectedExitException:
        print("Evaluacion detenida: la ventana de ViZDoom fue cerrada.")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evalua un checkpoint entrenado.")
    parser.add_argument(
        "--checkpoint",
        default="ppo_doom_recurrent",
        help="Nombre del checkpoint dentro de logs/checkpoints, sin .zip.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Cantidad maxima de pasos de evaluacion. Si se omite, corre hasta Ctrl+C o cierre de ventana.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(checkpoint=args.checkpoint, steps=args.steps)
