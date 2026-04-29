from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from doom_agent.config.schema import ProjectPaths, TrainingProfile
from doom_agent.shared.contracts import (
    CatalogSection,
    ProfileOverrides,
    ScenarioOverrides,
    TrainingProfilePayload,
)

DEFAULT_PROFILE_NAME = "default"
DEFAULT_SCENARIO_NAME = "basic"


@dataclass(frozen=True, slots=True)
class TrainingCatalog:
    defaults: ProfileOverrides
    profiles: CatalogSection
    scenarios: dict[str, ScenarioOverrides]


def build_project_paths(root_dir: Path | None = None) -> ProjectPaths:
    resolved_root = root_dir or Path(__file__).resolve().parents[3]
    artifacts_dir = resolved_root / "artifacts"
    legacy_logs_dir = resolved_root / "logs"
    checkpoints_dir = artifacts_dir / "checkpoints"
    config_dir = resolved_root / "configs"
    reports_dir = artifacts_dir / "reports"

    return ProjectPaths(
        root_dir=resolved_root,
        config_dir=config_dir,
        training_catalog_path=config_dir / "training_profiles.toml",
        data_dir=resolved_root / "data",
        scenarios_dir=resolved_root / "data" / "scenarios",
        artifacts_dir=artifacts_dir,
        checkpoints_dir=checkpoints_dir,
        auto_checkpoints_dir=checkpoints_dir / "auto",
        tensorboard_dir=artifacts_dir / "tensorboard",
        videos_dir=artifacts_dir / "videos",
        reports_dir=reports_dir,
        experiments_index_path=reports_dir / "index.json",
        legacy_logs_dir=legacy_logs_dir,
        legacy_checkpoints_dir=legacy_logs_dir / "checkpoints",
        legacy_auto_checkpoints_dir=legacy_logs_dir / "checkpoints" / "auto",
    )


def _merge_config_layers(*layers: ProfileOverrides) -> ProfileOverrides:
    merged: ProfileOverrides = {}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_config_layers(
                    cast(ProfileOverrides, merged[key]),
                    cast(ProfileOverrides, value),
                )
            else:
                merged[key] = value
    return merged


def _scenario_suffix(scenario_key: str) -> str:
    return "" if scenario_key == DEFAULT_SCENARIO_NAME else f"__{scenario_key}"


def _strip_scenario_suffix(name: str, scenario_key: str) -> str:
    suffix = _scenario_suffix(scenario_key)
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _apply_scenario_suffix(name: str, scenario_key: str) -> str:
    return f"{name}{_scenario_suffix(scenario_key)}"


def _require_str(config: ProfileOverrides, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise TypeError(f"Se esperaba una cadena para '{key}' y se obtuvo {type(value).__name__}.")
    return value


@lru_cache(maxsize=1)
def load_training_catalog() -> TrainingCatalog:
    project_paths = build_project_paths()
    config_path = project_paths.training_catalog_path
    raw_catalog = tomllib.loads(config_path.read_text(encoding="utf-8"))

    return TrainingCatalog(
        defaults=cast(ProfileOverrides, raw_catalog.get("defaults", {})),
        profiles=cast(CatalogSection, raw_catalog.get("profiles", {})),
        scenarios=cast(dict[str, ScenarioOverrides], raw_catalog.get("scenarios", {})),
    )


def _get_profile_settings(profile_name: str) -> ProfileOverrides:
    catalog = load_training_catalog()
    try:
        return catalog.profiles[profile_name]
    except KeyError as error:
        available = ", ".join(PROFILE_NAMES)
        raise ValueError(
            f"Perfil desconocido '{profile_name}'. Disponibles: {available}"
        ) from error


def _get_scenario_settings(scenario_key: str) -> ScenarioOverrides:
    catalog = load_training_catalog()
    try:
        return catalog.scenarios[scenario_key]
    except KeyError as error:
        available = ", ".join(SCENARIO_NAMES)
        raise ValueError(
            f"Escenario desconocido '{scenario_key}'. Disponibles: {available}"
        ) from error


def _build_runtime_names(
    checkpoint_name: str,
    tensorboard_run_name: str,
    scenario_key: str,
    previous_scenario_key: str | None = None,
) -> tuple[str, str]:
    base_checkpoint_name = checkpoint_name
    base_tensorboard_run_name = tensorboard_run_name

    if previous_scenario_key is not None:
        base_checkpoint_name = _strip_scenario_suffix(checkpoint_name, previous_scenario_key)
        base_tensorboard_run_name = _strip_scenario_suffix(
            tensorboard_run_name, previous_scenario_key
        )

    return (
        _apply_scenario_suffix(base_checkpoint_name, scenario_key),
        _apply_scenario_suffix(base_tensorboard_run_name, scenario_key),
    )


def _materialize_profile(
    profile_name: str,
    requested_timesteps: int | None = None,
    scenario_name: str | None = None,
    seed: int | None = None,
) -> TrainingProfile:
    catalog = load_training_catalog()
    profile_settings = _get_profile_settings(profile_name)
    default_scenario_key = profile_settings.get(
        "scenario_key",
        catalog.defaults.get("scenario_key", DEFAULT_SCENARIO_NAME),
    )
    if not isinstance(default_scenario_key, str):
        raise TypeError("El campo 'scenario_key' del catalogo debe ser una cadena.")
    scenario_key = scenario_name or default_scenario_key
    scenario_settings = _get_scenario_settings(scenario_key)

    merged = _merge_config_layers(catalog.defaults, profile_settings, scenario_settings)
    description = merged.pop("description", "")
    merged["scenario_key"] = scenario_key
    merged["scenario_description"] = description

    checkpoint_name, tensorboard_run_name = _build_runtime_names(
        checkpoint_name=_require_str(merged, "checkpoint_name"),
        tensorboard_run_name=_require_str(merged, "tensorboard_run_name"),
        scenario_key=scenario_key,
    )
    merged["checkpoint_name"] = checkpoint_name
    merged["tensorboard_run_name"] = tensorboard_run_name

    profile = TrainingProfile.from_dict(cast(TrainingProfilePayload, merged))
    if requested_timesteps is not None:
        profile = profile.with_timesteps(requested_timesteps)
    if seed is not None:
        profile = profile.with_seed(seed)
    return profile


def get_training_profile(
    profile_name: str = DEFAULT_PROFILE_NAME,
    requested_timesteps: int | None = None,
    scenario_name: str | None = None,
    seed: int | None = None,
) -> TrainingProfile:
    return _materialize_profile(
        profile_name=profile_name,
        requested_timesteps=requested_timesteps,
        scenario_name=scenario_name,
        seed=seed,
    )


def materialize_curriculum_profiles(
    profile_name: str = DEFAULT_PROFILE_NAME,
    requested_timesteps: int | None = None,
    seed: int | None = None,
) -> list[TrainingProfile]:
    base_profile = _materialize_profile(
        profile_name=profile_name,
        requested_timesteps=requested_timesteps,
        scenario_name=None,
        seed=seed,
    )
    if not base_profile.curriculum:
        return [base_profile]

    stages: list[TrainingProfile] = []
    for stage_index, stage in enumerate(base_profile.curriculum):
        _get_scenario_settings(stage.scenario_key)
        stage_profile = override_profile_scenario(base_profile, stage.scenario_key)
        if stage.requested_timesteps is not None:
            stage_profile = stage_profile.with_timesteps(stage.requested_timesteps)
        stage_profile = stage_profile.with_seed(base_profile.seed + stage_index)
        stages.append(stage_profile)
    return stages


def override_profile_scenario(profile: TrainingProfile, scenario_name: str) -> TrainingProfile:
    scenario_settings = _get_scenario_settings(scenario_name)
    profile_payload = profile.to_dict()
    profile_payload["checkpoint_name"], profile_payload["tensorboard_run_name"] = (
        _build_runtime_names(
            checkpoint_name=_require_str(
                cast(ProfileOverrides, profile_payload), "checkpoint_name"
            ),
            tensorboard_run_name=_require_str(
                cast(ProfileOverrides, profile_payload), "tensorboard_run_name"
            ),
            scenario_key=scenario_name,
            previous_scenario_key=profile.scenario_key,
        )
    )
    merged = _merge_config_layers(cast(ProfileOverrides, profile_payload), scenario_settings)
    merged["scenario_key"] = scenario_name
    merged["scenario_description"] = merged.pop("description", "")
    return TrainingProfile.from_dict(cast(TrainingProfilePayload, merged))


PROFILE_NAMES = tuple(sorted(load_training_catalog().profiles))
SCENARIO_NAMES = tuple(sorted(load_training_catalog().scenarios))
TRAINING_PROFILES = {
    profile_name: get_training_profile(profile_name) for profile_name in PROFILE_NAMES
}
