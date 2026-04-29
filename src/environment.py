"""Backward-compatible environment exports."""

from stable_baselines3.common.vec_env import VecEnv

from doom_agent.config import build_project_paths
from doom_agent.config.schema import TrainingProfile
from doom_agent.envs import DoomEnv, make_vectorized_env
from doom_agent.shared.contracts import LegacyConfigPayload, TrainingProfilePayload


def make_doom_env(config: LegacyConfigPayload, record: bool = False) -> VecEnv:
    profile_payload: LegacyConfigPayload = {
        "env_name": config["env_name"],
        "learning_rate": config.get("learning_rate", 1e-4),
        "n_steps": config.get("n_steps", 2048),
        "batch_size": config.get("batch_size", 64),
        "n_epochs": config.get("n_epochs", 10),
        "gamma": config.get("gamma", 0.99),
        "gae_lambda": config.get("gae_lambda", 0.95),
        "ent_coef": config.get("ent_coef", 0.01),
        "total_timesteps": config.get("total_timesteps", 500000),
        "render": config.get("render", False),
        "record": record or config.get("record", False),
        "save_name": config.get("save_name", "ppo_doom_recurrent"),
        "tb_log_name": config.get("tb_log_name", "PPO_LSTM_Doom"),
        "checkpoint_freq": config.get("checkpoint_freq", 50000),
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
    reward_shaping = config.get("reward_shaping")
    if reward_shaping is not None:
        profile_payload["reward_shaping"] = reward_shaping

    training_payload: TrainingProfilePayload = {
        "scenario_name": profile_payload["env_name"],
        "learning_rate": profile_payload.get("learning_rate", 1e-4),
        "n_steps": profile_payload.get("n_steps", 2048),
        "batch_size": profile_payload.get("batch_size", 64),
        "n_epochs": profile_payload.get("n_epochs", 10),
        "gamma": profile_payload.get("gamma", 0.99),
        "gae_lambda": profile_payload.get("gae_lambda", 0.95),
        "ent_coef": profile_payload.get("ent_coef", 0.01),
        "requested_timesteps": profile_payload.get("total_timesteps", 500000),
        "render": profile_payload.get("render", False),
        "record_video": profile_payload.get("record", False),
        "checkpoint_name": profile_payload.get("save_name", "ppo_doom_recurrent"),
        "tensorboard_run_name": profile_payload.get("tb_log_name", "PPO_LSTM_Doom"),
        "checkpoint_frequency": profile_payload.get("checkpoint_freq", 50000),
        "video_record_frequency": profile_payload.get(
            "video_record_frequency",
            profile_payload.get("checkpoint_freq", 50000),
        ),
        "video_length": profile_payload.get("video_length", 2000),
        "frame_stack": profile_payload.get("frame_stack", 4),
        "screen_width": profile_payload.get("screen_width", 84),
        "screen_height": profile_payload.get("screen_height", 84),
        "seed": profile_payload.get("seed", 42),
        "scenario_key": profile_payload.get("scenario_key", "basic"),
        "scenario_description": profile_payload.get("scenario_description", ""),
        "early_stopping": profile_payload.get("early_stopping", {}),
        "curriculum": profile_payload.get("curriculum", []),
    }
    if reward_shaping is not None:
        training_payload["reward_shaping"] = reward_shaping

    profile = TrainingProfile.from_dict(training_payload)
    return make_vectorized_env(profile, build_project_paths())


__all__ = ["DoomEnv", "make_doom_env", "make_vectorized_env"]
