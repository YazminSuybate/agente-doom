from __future__ import annotations

from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

from doom_agent.config.schema import EarlyStoppingConfig, ProjectPaths, TrainingProfile
from doom_agent.models import build_recurrent_ppo_model
from doom_agent.shared.contracts import EvaluationMetricsPayload
from doom_agent.utils.checkpoints import (
    ResolvedCheckpoint,
    build_checkpoint_metadata,
    checkpoint_zip_path,
    load_checkpoint_metadata,
    resolve_checkpoint,
    resolve_latest_checkpoint,
    save_checkpoint_bundle,
)

AUTO_RESUME_MODE = "auto"
LATEST_RESUME_MODE = "latest"


@dataclass(frozen=True, slots=True)
class ResumeState:
    mode: str
    checkpoint: ResolvedCheckpoint | None
    note: str | None = None

    @property
    def is_resumed(self) -> bool:
        return self.checkpoint is not None

    @property
    def resume_source(self) -> str | None:
        if self.checkpoint is None:
            return None
        return str(checkpoint_zip_path(self.checkpoint.checkpoint_stem))

    @property
    def resume_saved_timesteps(self) -> int | None:
        if self.checkpoint is None:
            return None
        return self.checkpoint.saved_timesteps


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    frequency: int
    episodes: int
    save_best: bool
    best_checkpoint_stem: Path


@dataclass(slots=True)
class EarlyStoppingTracker:
    config: EarlyStoppingConfig
    best_mean_reward: float = -inf
    evaluations_seen: int = 0
    no_improvement_evaluations: int = 0
    stopped: bool = False
    stop_reason: str | None = None

    def register(self, mean_reward: float) -> bool:
        improved = mean_reward > (self.best_mean_reward + self.config.min_delta)
        self.evaluations_seen += 1

        if improved:
            self.best_mean_reward = mean_reward
            self.no_improvement_evaluations = 0
        else:
            self.no_improvement_evaluations += 1

        if (
            self.config.enabled
            and self.evaluations_seen >= self.config.min_evaluations
            and self.no_improvement_evaluations >= self.config.patience_evaluations
        ):
            self.stopped = True
            self.stop_reason = (
                "Early stopping activado por falta de mejora en evaluacion: "
                f"{self.no_improvement_evaluations} evaluaciones sin superar "
                f"min_delta={self.config.min_delta}."
            )
        return improved


def _next_multiple(current_timesteps: int, frequency: int) -> int:
    return ((current_timesteps // frequency) + 1) * frequency


def _action_space_kind(action_space: gym.Space[Any]) -> str:
    if isinstance(action_space, gym.spaces.Discrete):
        return "discrete"
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return "multidiscrete"
    raise ValueError(f"Action space no soportado para reanudacion: {action_space}")


def _legacy_compatibility_issues(profile: TrainingProfile, checkpoint_stem: Path) -> list[str]:
    legacy_model = RecurrentPPO.load(str(checkpoint_zip_path(checkpoint_stem)))
    observation_shape = legacy_model.observation_space.shape
    issues: list[str] = []

    if observation_shape != (
        profile.frame_stack,
        profile.screen_height,
        profile.screen_width,
    ):
        issues.append(
            "La forma de observacion del checkpoint legacy "
            f"{observation_shape} no coincide con {(profile.frame_stack, profile.screen_height, profile.screen_width)}."
        )

    checkpoint_action_space_kind = _action_space_kind(legacy_model.action_space)
    if checkpoint_action_space_kind != profile.action_space_kind:
        issues.append(
            f"El action space del checkpoint legacy ({checkpoint_action_space_kind}) "
            f"no coincide con el actual ({profile.action_space_kind})."
        )
    return issues


def resolve_resume_state(
    project_paths: ProjectPaths,
    profile: TrainingProfile,
    *,
    resume_mode: str = AUTO_RESUME_MODE,
    from_scratch: bool = False,
    allow_scenario_change: bool = False,
) -> ResumeState:
    if from_scratch:
        return ResumeState(mode="from_scratch", checkpoint=None)

    if resume_mode in {AUTO_RESUME_MODE, LATEST_RESUME_MODE}:
        checkpoint = resolve_latest_checkpoint(project_paths, profile.checkpoint_name)
        if checkpoint is None:
            return ResumeState(mode=resume_mode, checkpoint=None)
    else:
        checkpoint = resolve_checkpoint(project_paths, resume_mode)

    if checkpoint.metadata is None:
        issues = _legacy_compatibility_issues(profile, checkpoint.checkpoint_stem)
        if issues:
            formatted_issues = "\n".join(f"- {issue}" for issue in issues)
            raise ValueError(
                "El checkpoint legacy no es compatible con el entrenamiento actual:\n"
                f"{formatted_issues}"
            )
        note = (
            "Checkpoint legacy sin metadata completa. Se valido observation/action space, "
            "pero no fue posible comparar todos los hiperparametros."
        )
        return ResumeState(mode=resume_mode, checkpoint=checkpoint, note=note)

    previous_profile = TrainingProfile.from_dict(checkpoint.metadata["profile"])
    if allow_scenario_change:
        issues = profile.model_compatibility_issues(previous_profile)
    else:
        issues = profile.resume_compatibility_issues(previous_profile)
    if issues:
        formatted_issues = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(
            "El checkpoint encontrado no es compatible con el perfil actual. "
            "Usa '--from-scratch' o ajusta el perfil.\n"
            f"{formatted_issues}"
        )

    return ResumeState(mode=resume_mode, checkpoint=checkpoint)


def load_training_model(
    env: VecEnv,
    profile: TrainingProfile,
    tensorboard_dir: Path,
    resume_state: ResumeState,
) -> RecurrentPPO:
    if not resume_state.is_resumed:
        return build_recurrent_ppo_model(env, profile, str(tensorboard_dir))

    assert resume_state.checkpoint is not None
    model = RecurrentPPO.load(
        str(checkpoint_zip_path(resume_state.checkpoint.checkpoint_stem)),
        env=env,
    )
    model.set_random_seed(profile.seed)
    model.tensorboard_log = str(tensorboard_dir)
    return model


def load_best_mean_reward(best_checkpoint_stem: Path) -> float:
    metadata = load_checkpoint_metadata(best_checkpoint_stem)
    if metadata is None:
        return -inf

    evaluation_metrics = metadata["evaluation_metrics"]
    if evaluation_metrics is None:
        return -inf

    return float(evaluation_metrics["mean_reward"])


class PeriodicTrainingCallback(BaseCallback):
    def __init__(
        self,
        *,
        profile_name: str,
        profile: TrainingProfile,
        auto_checkpoint_dir: Path,
        evaluation_settings: EvaluationSettings,
        eval_env: VecEnv,
        resume_state: ResumeState,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.profile_name = profile_name
        self.profile = profile
        self.auto_checkpoint_dir = auto_checkpoint_dir
        self.evaluation_settings = evaluation_settings
        self.eval_env = eval_env
        self.resume_state = resume_state
        self.next_checkpoint_step = profile.checkpoint_frequency
        self.next_eval_step = evaluation_settings.frequency
        self.best_mean_reward = load_best_mean_reward(evaluation_settings.best_checkpoint_stem)
        self.early_stopping = EarlyStoppingTracker(
            config=profile.early_stopping,
            best_mean_reward=self.best_mean_reward,
        )
        self.last_evaluation_metrics: EvaluationMetricsPayload | None = None

    def _on_training_start(self) -> None:
        current_timesteps = self.model.num_timesteps
        self.next_checkpoint_step = _next_multiple(
            current_timesteps, self.profile.checkpoint_frequency
        )
        self.next_eval_step = _next_multiple(current_timesteps, self.evaluation_settings.frequency)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval_step:
            self._run_periodic_evaluation()
            if self.early_stopping.stopped:
                return False
            self.next_eval_step += self.evaluation_settings.frequency

        if self.num_timesteps >= self.next_checkpoint_step:
            self._save_periodic_checkpoint()
            self.next_checkpoint_step += self.profile.checkpoint_frequency
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()

    def _run_periodic_evaluation(self) -> None:
        rewards, lengths = cast(
            tuple[list[float], list[int]],
            evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.evaluation_settings.episodes,
                deterministic=True,
                return_episode_rewards=True,
                warn=False,
            ),
        )
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_length = float(np.mean(lengths))
        self.last_evaluation_metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_episode_length": mean_length,
            "episodes": self.evaluation_settings.episodes,
        }

        if self.verbose:
            print(
                "Evaluacion periodica: "
                f"mean_reward={mean_reward:.3f}, std_reward={std_reward:.3f}, "
                f"mean_length={mean_length:.1f}."
            )

        improved = self.early_stopping.register(mean_reward)
        self.best_mean_reward = self.early_stopping.best_mean_reward

        if self.verbose and self.early_stopping.stopped and self.early_stopping.stop_reason:
            print(self.early_stopping.stop_reason)

        if self.evaluation_settings.save_best and improved:
            self.best_mean_reward = mean_reward
            metadata = build_checkpoint_metadata(
                profile_name=self.profile_name,
                profile=self.profile,
                saved_timesteps=self.num_timesteps,
                resume_source=self.resume_state.resume_source,
                resume_saved_timesteps=self.resume_state.resume_saved_timesteps,
                training_status="best_model",
                evaluation_metrics=self.last_evaluation_metrics,
                is_best_checkpoint=True,
            )
            save_checkpoint_bundle(
                self.model,
                self.evaluation_settings.best_checkpoint_stem,
                metadata,
            )
            if self.verbose:
                print(
                    "Nuevo mejor modelo guardado en "
                    f"{self.evaluation_settings.best_checkpoint_stem.with_suffix('.zip')}"
                )

    def _save_periodic_checkpoint(self) -> None:
        checkpoint_stem = self.auto_checkpoint_dir / (
            f"{self.profile.checkpoint_name}_{self.num_timesteps}_steps"
        )
        metadata = build_checkpoint_metadata(
            profile_name=self.profile_name,
            profile=self.profile,
            saved_timesteps=self.num_timesteps,
            resume_source=self.resume_state.resume_source,
            resume_saved_timesteps=self.resume_state.resume_saved_timesteps,
            training_status="periodic_checkpoint",
            evaluation_metrics=self.last_evaluation_metrics,
        )
        save_checkpoint_bundle(self.model, checkpoint_stem, metadata)
        if self.verbose:
            print(f"Checkpoint guardado en {checkpoint_stem.with_suffix('.zip')}")
