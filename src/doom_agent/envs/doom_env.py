from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import gymnasium as gym
import numpy as np
import vizdoom as zd
from gymnasium.core import RenderFrame
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecMonitor,
    VecVideoRecorder,
)

from doom_agent.config.schema import ProjectPaths, TrainingProfile
from doom_agent.envs.reward import RewardShaper
from doom_agent.shared.types import BinaryAction, Observation
from doom_agent.utils.filesystem import ensure_directories


def normalize_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] != 3:
        return np.moveaxis(frame, 0, -1)
    return frame


def preprocess_frame(frame: np.ndarray, width: int, height: int) -> Observation:
    frame = normalize_rgb_frame(frame)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized.reshape(1, height, width).astype(np.uint8)


class DoomEnv(gym.Env[Observation, BinaryAction]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}

    def __init__(
        self,
        game: zd.DoomGame,
        observation_width: int,
        observation_height: int,
        action_space_kind: str,
        render_mode: str,
        reward_shaper: RewardShaper,
    ) -> None:
        super().__init__()
        self.game = game
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.button_count = self.game.get_available_buttons_size()
        self.action_space_kind = action_space_kind
        self.render_mode = render_mode
        self.reward_shaper = reward_shaper
        self.action_space: gym.Space[Any]

        if self.action_space_kind == "multidiscrete":
            self.action_space = gym.spaces.MultiDiscrete(
                np.full(self.button_count, 2, dtype=np.int64)
            )
        else:
            self.action_space = gym.spaces.Discrete(self.button_count)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, self.observation_height, self.observation_width),
            dtype=np.uint8,
        )

    def step(
        self,
        action: BinaryAction,
    ) -> tuple[Observation, float, bool, bool, dict[str, float]]:
        binary_action = self._normalize_action(action)
        raw_reward = float(self.game.make_action(binary_action.tolist()))
        reward = self.reward_shaper.apply(raw_reward)
        state = self.game.get_state()
        terminated = self.game.is_episode_finished()
        truncated = False
        observation = self._observation_from_state(state)
        info = {"raw_reward": raw_reward, "shaped_reward": reward}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[Observation, dict[str, object]]:
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        return self._observation_from_state(state), {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        state = cast(zd.GameState | None, self.game.get_state())
        if state is None:
            return cast(RenderFrame, np.zeros((240, 320, 3), dtype=np.uint8))
        return cast(RenderFrame, normalize_rgb_frame(state.screen_buffer))

    def close(self) -> None:
        self.game.close()

    def _normalize_action(self, action: BinaryAction) -> BinaryAction:
        if self.action_space_kind == "discrete":
            discrete_action = int(np.asarray(action).item())
            if discrete_action < 0 or discrete_action >= self.button_count:
                raise ValueError(
                    f"La accion discreta debe estar entre 0 y {self.button_count - 1}."
                )
            encoded_action = np.zeros(self.button_count, dtype=np.int32)
            encoded_action[discrete_action] = 1
            return encoded_action

        normalized_action = np.asarray(action, dtype=np.int32).reshape(-1)
        if normalized_action.size != self.button_count:
            raise ValueError(
                f"Se esperaban {self.button_count} botones y se recibieron {normalized_action.size}."
            )
        return np.clip(normalized_action, 0, 1)

    def _observation_from_state(self, state: zd.GameState | None) -> Observation:
        if state is None:
            shape = cast(tuple[int, int, int], self.observation_space.shape)
            return np.zeros(shape, dtype=np.uint8)
        return preprocess_frame(
            state.screen_buffer,
            width=self.observation_width,
            height=self.observation_height,
        )


def build_doom_game(profile: TrainingProfile, project_paths: ProjectPaths) -> zd.DoomGame:
    scenario_path = profile.scenario_path(project_paths)
    game = zd.DoomGame()
    game.load_config(str(Path(scenario_path)))
    game.set_seed(profile.seed)
    game.set_window_visible(profile.render)
    game.set_screen_format(zd.ScreenFormat.RGB24)
    game.set_screen_resolution(zd.ScreenResolution.RES_320X240)
    game.init()
    return game


def should_record_video(step: int, frequency: int) -> bool:
    return step == 0 or step % frequency == 0


def make_vectorized_env(profile: TrainingProfile, project_paths: ProjectPaths) -> VecEnv:
    def _build_env() -> DoomEnv:
        game = build_doom_game(profile, project_paths)
        return DoomEnv(
            game=game,
            observation_width=profile.screen_width,
            observation_height=profile.screen_height,
            action_space_kind=profile.action_space_kind,
            render_mode="rgb_array",
            reward_shaper=RewardShaper(profile.reward_shaping),
        )

    env: VecEnv = DummyVecEnv([_build_env])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=profile.frame_stack)

    if profile.record_video:
        ensure_directories([project_paths.videos_dir])
        env = VecVideoRecorder(
            venv=env,
            video_folder=str(project_paths.videos_dir),
            record_video_trigger=lambda step: should_record_video(
                step, profile.video_record_frequency
            ),
            video_length=profile.video_length,
            name_prefix=profile.checkpoint_name,
        )
    return env
