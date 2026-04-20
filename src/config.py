import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS_DIR = os.path.join(BASE_DIR, 'data', 'scenarios')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
VIDEO_DIR = os.path.join(BASE_DIR, 'data', 'videos')

CONFIG = {
    "env_name": "basic.cfg", 
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "total_timesteps": 500000,
    "render": True,
    "record": True,
    "save_name": "ppo_doom_recurrent",
    "tb_log_name": "PPO_LSTM_Doom",
    "checkpoint_freq": 50000
}

FAST_CONFIG = {
    **CONFIG,
    "n_steps": 256,
    "n_epochs": 3,
    "total_timesteps": 10000,
    "render": False,
    "record": False,
    "save_name": "ppo_doom_recurrent_fast",
    "tb_log_name": "PPO_LSTM_Doom_Fast",
    "checkpoint_freq": 2000
}

EFFICIENT_CONFIG = {
    **CONFIG,
    "n_steps": 1024,
    "n_epochs": 5,
    "total_timesteps": 100000,
    "render": False,
    "record": False,
    "save_name": "ppo_doom_recurrent_efficient",
    "tb_log_name": "PPO_LSTM_Doom_Efficient",
    "checkpoint_freq": 10000
}

TRAINING_CONFIGS = {
    "default": CONFIG,
    "efficient": EFFICIENT_CONFIG,
    "fast": FAST_CONFIG
}
