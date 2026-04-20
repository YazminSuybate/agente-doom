import os
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from environment import make_doom_env
from config import CONFIG, LOG_DIR, SCENARIOS_DIR

def evaluate():
    config_eval = {
        **CONFIG, 
        "SCENARIOS_DIR": SCENARIOS_DIR, 
        "render": True 
    }
    
    env = make_doom_env(config_eval)
    
    model_path = os.path.join(LOG_DIR, "checkpoints", "ppo_doom_recurrent")
    if not os.path.exists(model_path + ".zip"):
        print("¡Error! No existe un modelo entrenado en la ruta especificada.")
        return

    model = RecurrentPPO.load(model_path, env=env)
    
    obs = env.reset()
    lstm_states = None
    episode_starts = [True]

    print("Evaluando agente... Presiona Ctrl+C para salir.")
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = dones
        env.render()

if __name__ == "__main__":
    evaluate()