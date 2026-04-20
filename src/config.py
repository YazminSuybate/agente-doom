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
    "render": True  
}