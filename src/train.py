from config import CONFIG, LOG_DIR, SCENARIOS_DIR, VIDEO_DIR
from environment import make_doom_env
from model import get_recurrent_model
import os

def train():
    for p in [LOG_DIR, VIDEO_DIR]: os.makedirs(p, exist_ok=True)
    
    config_extended = {
        **CONFIG, 
        "SCENARIOS_DIR": SCENARIOS_DIR, 
        "VIDEO_DIR": VIDEO_DIR,
        "LOG_DIR": LOG_DIR 
    }
    
    env = make_doom_env(config_extended, record=True)
    model = get_recurrent_model(env, config_extended)
    
    print(f"Iniciando entrenamiento en {CONFIG['env_name']} con LSTM...")
    model.learn(total_timesteps=CONFIG['total_timesteps'], tb_log_name="PPO_LSTM_Doom")
    
    save_path = os.path.join(LOG_DIR, "checkpoints", "ppo_doom_recurrent")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Entrenamiento completado. Modelo guardado en {save_path}")

if __name__ == "__main__":
    train()