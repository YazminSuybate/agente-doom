from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, TypedDict


class RewardShapingPayload(TypedDict, total=False):
    scale: float
    offset: float
    clip_min: float | None
    clip_max: float | None


class EarlyStoppingPayload(TypedDict, total=False):
    enabled: bool
    patience_evaluations: int
    min_evaluations: int
    min_delta: float


class CurriculumStagePayload(TypedDict, total=False):
    scenario_key: str
    requested_timesteps: int | None


class TrainingProfilePayload(TypedDict, total=False):
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
    frame_stack: int
    screen_width: int
    screen_height: int
    seed: int
    action_space_kind: str
    scenario_key: str
    scenario_description: str
    reward_shaping: RewardShapingPayload
    early_stopping: EarlyStoppingPayload
    curriculum: list[CurriculumStagePayload]
    effective_timesteps: int
    uses_rounded_timesteps: bool


class EvaluationMetricsPayload(TypedDict):
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    episodes: int


class CheckpointMetadataPayload(TypedDict):
    profile_name: str
    saved_timesteps: int
    saved_at_utc: str
    profile: TrainingProfilePayload
    resume_source: str | None
    resume_saved_timesteps: int | None
    training_status: str | None
    evaluation_metrics: EvaluationMetricsPayload | None
    is_best_checkpoint: bool


class TrainingRunReportPayload(TypedDict):
    run_id: str
    created_at_utc: str
    profile_name: str
    run_label: str | None
    scenario_key: str
    scenario_name: str
    checkpoint_name: str
    checkpoint_path: str
    best_checkpoint_path: str | None
    training_status: str
    completed: bool
    requested_timesteps: int
    effective_timesteps: int
    saved_timesteps: int
    seed: int
    resume_mode: str
    resume_source: str | None
    resume_saved_timesteps: int | None
    evaluation_metrics: EvaluationMetricsPayload | None
    duration_seconds: float
    stopped_early: bool
    stop_reason: str | None


class ExperimentIndexEntryPayload(TypedDict):
    run_id: str
    created_at_utc: str
    profile_name: str
    run_label: str | None
    scenario_key: str
    checkpoint_name: str
    checkpoint_path: str
    best_checkpoint_path: str | None
    saved_timesteps: int
    training_status: str
    mean_reward: float | None
    seed: int
    report_path: str


class ExperimentIndexPayload(TypedDict):
    runs: list[ExperimentIndexEntryPayload]


class LegacyConfigPayload(TypedDict, total=False):
    env_name: str
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    total_timesteps: int
    requested_timesteps: int
    effective_timesteps: int
    render: bool
    record: bool
    save_name: str
    tb_log_name: str
    checkpoint_freq: int
    video_record_frequency: int
    video_length: int
    frame_stack: int
    screen_width: int
    screen_height: int
    seed: int
    scenario_key: str
    scenario_description: str
    reward_shaping: RewardShapingPayload
    early_stopping: EarlyStoppingPayload
    curriculum: list[CurriculumStagePayload]
    LOG_DIR: str
    SCENARIOS_DIR: str
    VIDEO_DIR: str


class SaveableModel(Protocol):
    def save(self, path: str) -> None: ...


ProfileOverrides: TypeAlias = dict[str, object]
ScenarioOverrides: TypeAlias = dict[str, object]
CatalogSection: TypeAlias = dict[str, ProfileOverrides]
CheckpointSelection: TypeAlias = Literal["best", "last", "exact"]
