from __future__ import annotations

from typing import cast

import gymnasium as gym
import torch as th
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv

from doom_agent.config.schema import TrainingProfile
from doom_agent.shared.types import Observation


class DoomFeatureExtractor(BaseFeaturesExtractor):
    """CNN extractor tuned for stacked grayscale observations."""

    def __init__(self, observation_space: gym.Space[Observation], features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        shape = cast(tuple[int, ...], observation_space.shape)
        input_channels = shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            flattened_size = self.cnn(sample).shape[1]

        self.projection = nn.Sequential(
            nn.Linear(flattened_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return cast(th.Tensor, self.projection(self.cnn(observations)))


def build_recurrent_ppo_model(
    env: VecEnv,
    profile: TrainingProfile,
    tensorboard_log_dir: str,
) -> RecurrentPPO:
    return RecurrentPPO(
        policy="CnnLstmPolicy",
        env=env,
        learning_rate=profile.learning_rate,
        n_steps=profile.n_steps,
        batch_size=profile.batch_size,
        n_epochs=profile.n_epochs,
        gamma=profile.gamma,
        gae_lambda=profile.gae_lambda,
        ent_coef=profile.ent_coef,
        seed=profile.seed,
        tensorboard_log=tensorboard_log_dir,
        verbose=1,
        policy_kwargs={"features_extractor_class": DoomFeatureExtractor},
    )
