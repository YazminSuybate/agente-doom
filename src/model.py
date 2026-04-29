"""Backward-compatible model exports."""

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv

from doom_agent.config.schema import TrainingProfile
from doom_agent.models import DoomFeatureExtractor, build_recurrent_ppo_model
from doom_agent.shared.contracts import LegacyConfigPayload, TrainingProfilePayload


def _legacy_config_to_profile(config: LegacyConfigPayload) -> TrainingProfile:
    reward_shaping = config.get("reward_shaping")
    training_payload: TrainingProfilePayload = {
        "scenario_name": config.get("env_name", "basic.cfg"),
        "learning_rate": config.get("learning_rate", 1e-4),
        "n_steps": config.get("n_steps", 2048),
        "batch_size": config.get("batch_size", 64),
        "n_epochs": config.get("n_epochs", 10),
        "gamma": config.get("gamma", 0.99),
        "gae_lambda": config.get("gae_lambda", 0.95),
        "ent_coef": config.get("ent_coef", 0.01),
        "requested_timesteps": config.get(
            "total_timesteps", config.get("requested_timesteps", 500000)
        ),
        "render": config.get("render", False),
        "record_video": config.get("record", False),
        "checkpoint_name": config.get("save_name", "ppo_doom_recurrent"),
        "tensorboard_run_name": config.get("tb_log_name", "PPO_LSTM_Doom"),
        "checkpoint_frequency": config.get("checkpoint_freq", 50000),
        "video_record_frequency": config.get(
            "video_record_frequency",
            config.get("checkpoint_freq", 50000),
        ),
        "video_length": config.get("video_length", 2000),
        "frame_stack": config.get("frame_stack", 4),
        "screen_width": config.get("screen_width", 84),
        "screen_height": config.get("screen_height", 84),
        "seed": config.get("seed", 42),
        "scenario_key": config.get("scenario_key", "basic"),
        "scenario_description": config.get("scenario_description", ""),
        "early_stopping": config.get("early_stopping", {}),
        "curriculum": config.get("curriculum", []),
    }
    if reward_shaping is not None:
        training_payload["reward_shaping"] = reward_shaping

    return TrainingProfile.from_dict(training_payload)


def get_recurrent_model(env: VecEnv, config: LegacyConfigPayload) -> RecurrentPPO:
    profile = _legacy_config_to_profile(config)
    tensorboard_log_dir = config.get("LOG_DIR", "./artifacts/tensorboard")
    return build_recurrent_ppo_model(env, profile, tensorboard_log_dir)


DoomCNN = DoomFeatureExtractor

__all__ = ["DoomCNN", "DoomFeatureExtractor", "build_recurrent_ppo_model", "get_recurrent_model"]
