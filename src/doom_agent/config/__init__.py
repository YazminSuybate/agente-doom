"""Configuration models and training catalog accessors."""

from doom_agent.config.profiles import (
    DEFAULT_PROFILE_NAME,
    DEFAULT_SCENARIO_NAME,
    PROFILE_NAMES,
    SCENARIO_NAMES,
    TRAINING_PROFILES,
    build_project_paths,
    get_training_profile,
    load_training_catalog,
    materialize_curriculum_profiles,
    override_profile_scenario,
)
from doom_agent.config.schema import (
    CurriculumStageConfig,
    EarlyStoppingConfig,
    ProjectPaths,
    RewardShapingConfig,
    TrainingProfile,
)

__all__ = [
    "DEFAULT_PROFILE_NAME",
    "DEFAULT_SCENARIO_NAME",
    "CurriculumStageConfig",
    "EarlyStoppingConfig",
    "PROFILE_NAMES",
    "ProjectPaths",
    "RewardShapingConfig",
    "SCENARIO_NAMES",
    "TRAINING_PROFILES",
    "TrainingProfile",
    "build_project_paths",
    "get_training_profile",
    "load_training_catalog",
    "materialize_curriculum_profiles",
    "override_profile_scenario",
]
