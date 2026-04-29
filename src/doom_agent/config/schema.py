from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil
from pathlib import Path

from doom_agent.shared.contracts import (
    CurriculumStagePayload,
    EarlyStoppingPayload,
    RewardShapingPayload,
    TrainingProfilePayload,
)


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root_dir: Path
    config_dir: Path
    training_catalog_path: Path
    data_dir: Path
    scenarios_dir: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    auto_checkpoints_dir: Path
    tensorboard_dir: Path
    videos_dir: Path
    reports_dir: Path
    experiments_index_path: Path
    legacy_logs_dir: Path
    legacy_checkpoints_dir: Path
    legacy_auto_checkpoints_dir: Path


@dataclass(frozen=True, slots=True)
class RewardShapingConfig:
    scale: float = 1.0
    offset: float = 0.0
    clip_min: float | None = None
    clip_max: float | None = None

    def apply(self, reward: float) -> float:
        shaped_reward = (reward + self.offset) * self.scale
        if self.clip_min is not None and shaped_reward < self.clip_min:
            shaped_reward = self.clip_min
        if self.clip_max is not None and shaped_reward > self.clip_max:
            shaped_reward = self.clip_max
        return float(shaped_reward)

    def validate(self) -> None:
        if self.scale <= 0:
            raise ValueError("'reward_shaping.scale' debe ser mayor que cero.")

        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min > self.clip_max
        ):
            raise ValueError("'reward_shaping.clip_min' no puede ser mayor que 'clip_max'.")

    def to_dict(self) -> RewardShapingPayload:
        return {
            "scale": self.scale,
            "offset": self.offset,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    @classmethod
    def from_dict(cls, payload: RewardShapingPayload | None) -> RewardShapingConfig:
        if payload is None:
            return cls()
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    enabled: bool = False
    patience_evaluations: int = 3
    min_evaluations: int = 2
    min_delta: float = 0.0

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.patience_evaluations <= 0:
            raise ValueError("'early_stopping.patience_evaluations' debe ser mayor que cero.")
        if self.min_evaluations <= 0:
            raise ValueError("'early_stopping.min_evaluations' debe ser mayor que cero.")
        if self.min_delta < 0:
            raise ValueError("'early_stopping.min_delta' no puede ser negativo.")

    def to_dict(self) -> EarlyStoppingPayload:
        return {
            "enabled": self.enabled,
            "patience_evaluations": self.patience_evaluations,
            "min_evaluations": self.min_evaluations,
            "min_delta": self.min_delta,
        }

    @classmethod
    def from_dict(cls, payload: EarlyStoppingPayload | None) -> EarlyStoppingConfig:
        if payload is None:
            return cls()
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class CurriculumStageConfig:
    scenario_key: str
    requested_timesteps: int | None = None

    def validate(self) -> None:
        if not self.scenario_key:
            raise ValueError("'curriculum.scenario_key' no puede estar vacio.")
        if self.requested_timesteps is not None and self.requested_timesteps <= 0:
            raise ValueError("'curriculum.requested_timesteps' debe ser mayor que cero.")

    def to_dict(self) -> CurriculumStagePayload:
        return {
            "scenario_key": self.scenario_key,
            "requested_timesteps": self.requested_timesteps,
        }

    @classmethod
    def from_dict(cls, payload: CurriculumStagePayload) -> CurriculumStageConfig:
        return cls(
            scenario_key=payload["scenario_key"],
            requested_timesteps=payload.get("requested_timesteps"),
        )


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    scenario_name: str
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    requested_timesteps: int
    render: bool
    record_video: bool
    checkpoint_name: str
    tensorboard_run_name: str
    checkpoint_frequency: int
    video_record_frequency: int
    video_length: int
    frame_stack: int = 4
    screen_width: int = 84
    screen_height: int = 84
    seed: int = 42
    action_space_kind: str = "multidiscrete"
    scenario_key: str = "basic"
    scenario_description: str = ""
    reward_shaping: RewardShapingConfig = field(default_factory=RewardShapingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    curriculum: tuple[CurriculumStageConfig, ...] = ()

    @property
    def effective_timesteps(self) -> int:
        return ceil(self.requested_timesteps / self.n_steps) * self.n_steps

    @property
    def uses_rounded_timesteps(self) -> bool:
        return self.effective_timesteps != self.requested_timesteps

    def with_timesteps(self, requested_timesteps: int) -> TrainingProfile:
        return replace(self, requested_timesteps=requested_timesteps)

    def with_seed(self, seed: int) -> TrainingProfile:
        return replace(self, seed=seed)

    def with_names(self, checkpoint_name: str, tensorboard_run_name: str) -> TrainingProfile:
        return replace(
            self,
            checkpoint_name=checkpoint_name,
            tensorboard_run_name=tensorboard_run_name,
        )

    def for_evaluation(self, render: bool = True) -> TrainingProfile:
        return replace(self, render=render, record_video=False)

    def scenario_path(self, paths: ProjectPaths) -> Path:
        return paths.scenarios_dir / self.scenario_name

    def validate(self, paths: ProjectPaths) -> None:
        positive_fields = {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "requested_timesteps": self.requested_timesteps,
            "checkpoint_frequency": self.checkpoint_frequency,
            "video_record_frequency": self.video_record_frequency,
            "video_length": self.video_length,
            "frame_stack": self.frame_stack,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "seed": self.seed,
        }
        for field_name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"'{field_name}' debe ser mayor que cero.")

        if self.batch_size > self.n_steps:
            raise ValueError("'batch_size' no puede ser mayor que 'n_steps'.")

        if not 0 <= self.ent_coef:
            raise ValueError("'ent_coef' no puede ser negativo.")

        if not 0 < self.gamma <= 1:
            raise ValueError("'gamma' debe estar entre 0 y 1.")

        if not 0 < self.gae_lambda <= 1:
            raise ValueError("'gae_lambda' debe estar entre 0 y 1.")

        if self.action_space_kind not in {"discrete", "multidiscrete"}:
            raise ValueError("'action_space_kind' debe ser 'discrete' o 'multidiscrete'.")

        if not self.scenario_key:
            raise ValueError("'scenario_key' no puede estar vacio.")

        self.reward_shaping.validate()
        self.early_stopping.validate()
        for stage in self.curriculum:
            stage.validate()

        scenario_path = self.scenario_path(paths)
        if not scenario_path.exists():
            raise FileNotFoundError(f"No existe el escenario: {scenario_path}")

    def to_dict(self) -> TrainingProfilePayload:
        return {
            "scenario_name": self.scenario_name,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "requested_timesteps": self.requested_timesteps,
            "render": self.render,
            "record_video": self.record_video,
            "checkpoint_name": self.checkpoint_name,
            "tensorboard_run_name": self.tensorboard_run_name,
            "checkpoint_frequency": self.checkpoint_frequency,
            "video_record_frequency": self.video_record_frequency,
            "video_length": self.video_length,
            "frame_stack": self.frame_stack,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "seed": self.seed,
            "action_space_kind": self.action_space_kind,
            "scenario_key": self.scenario_key,
            "scenario_description": self.scenario_description,
            "reward_shaping": self.reward_shaping.to_dict(),
            "early_stopping": self.early_stopping.to_dict(),
            "curriculum": [stage.to_dict() for stage in self.curriculum],
            "effective_timesteps": self.effective_timesteps,
            "uses_rounded_timesteps": self.uses_rounded_timesteps,
        }

    def resume_compatibility_signature(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "scenario_key": self.scenario_key,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "frame_stack": self.frame_stack,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "seed": self.seed,
            "action_space_kind": self.action_space_kind,
            "reward_shaping": self.reward_shaping.to_dict(),
        }

    def model_compatibility_signature(self) -> dict[str, object]:
        return {
            "frame_stack": self.frame_stack,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "action_space_kind": self.action_space_kind,
        }

    def resume_compatibility_issues(self, previous_profile: TrainingProfile) -> list[str]:
        current_signature = self.resume_compatibility_signature()
        previous_signature = previous_profile.resume_compatibility_signature()
        issues: list[str] = []

        for key, current_value in current_signature.items():
            previous_value = previous_signature[key]
            if current_value != previous_value:
                issues.append(
                    f"'{key}' difiere entre el perfil actual ({current_value}) "
                    f"y el checkpoint ({previous_value})."
                )
        return issues

    def model_compatibility_issues(self, previous_profile: TrainingProfile) -> list[str]:
        current_signature = self.model_compatibility_signature()
        previous_signature = previous_profile.model_compatibility_signature()
        issues: list[str] = []

        for key, current_value in current_signature.items():
            previous_value = previous_signature[key]
            if current_value != previous_value:
                issues.append(
                    f"'{key}' difiere entre el perfil actual ({current_value}) "
                    f"y el checkpoint ({previous_value})."
                )
        return issues

    @classmethod
    def from_dict(cls, payload: TrainingProfilePayload) -> TrainingProfile:
        return cls(
            scenario_name=payload["scenario_name"],
            learning_rate=payload["learning_rate"],
            n_steps=payload["n_steps"],
            batch_size=payload["batch_size"],
            n_epochs=payload["n_epochs"],
            gamma=payload["gamma"],
            gae_lambda=payload["gae_lambda"],
            ent_coef=payload["ent_coef"],
            requested_timesteps=payload["requested_timesteps"],
            render=payload["render"],
            record_video=payload["record_video"],
            checkpoint_name=payload["checkpoint_name"],
            tensorboard_run_name=payload["tensorboard_run_name"],
            checkpoint_frequency=payload["checkpoint_frequency"],
            video_record_frequency=payload["video_record_frequency"],
            video_length=payload["video_length"],
            frame_stack=payload.get("frame_stack", 4),
            screen_width=payload.get("screen_width", 84),
            screen_height=payload.get("screen_height", 84),
            seed=payload.get("seed", 42),
            action_space_kind=payload.get("action_space_kind", "multidiscrete"),
            scenario_key=payload.get("scenario_key", "basic"),
            scenario_description=payload.get("scenario_description", ""),
            reward_shaping=RewardShapingConfig.from_dict(payload.get("reward_shaping")),
            early_stopping=EarlyStoppingConfig.from_dict(payload.get("early_stopping")),
            curriculum=tuple(
                CurriculumStageConfig.from_dict(stage) for stage in payload.get("curriculum", [])
            ),
        )
