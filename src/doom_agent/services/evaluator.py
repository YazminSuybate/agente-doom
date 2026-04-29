from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
from sb3_contrib import RecurrentPPO
from vizdoom import ViZDoomUnexpectedExitException

from doom_agent.config import (
    DEFAULT_PROFILE_NAME,
    build_project_paths,
    get_training_profile,
    override_profile_scenario,
)
from doom_agent.config.schema import TrainingProfile
from doom_agent.envs import make_vectorized_env
from doom_agent.shared.contracts import CheckpointSelection
from doom_agent.utils.checkpoints import (
    checkpoint_zip_path,
    load_checkpoint_metadata,
    resolve_checkpoint_preference,
)


def infer_legacy_profile(checkpoint_stem: Path) -> TrainingProfile:
    model = RecurrentPPO.load(str(checkpoint_zip_path(checkpoint_stem)))
    action_space = model.action_space
    base_profile = get_training_profile(DEFAULT_PROFILE_NAME).for_evaluation()

    if isinstance(action_space, gym.spaces.Discrete):
        return replace(base_profile, action_space_kind="discrete")

    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return replace(base_profile, action_space_kind="multidiscrete")

    raise ValueError(f"No se pudo inferir un action space compatible: {action_space}")


def resolve_profile_for_evaluation(
    checkpoint_name: str | None = None,
    profile_name: str = DEFAULT_PROFILE_NAME,
    checkpoint_selection: CheckpointSelection = "best",
    scenario_name: str | None = None,
) -> tuple[TrainingProfile, Path]:
    project_paths = build_project_paths()
    resolved_checkpoint_name = checkpoint_name
    if resolved_checkpoint_name is None:
        resolved_checkpoint_name = get_training_profile(
            profile_name,
            scenario_name=scenario_name,
        ).checkpoint_name

    resolved_checkpoint = resolve_checkpoint_preference(
        project_paths,
        resolved_checkpoint_name,
        preference=checkpoint_selection,
    )
    checkpoint_stem = resolved_checkpoint.checkpoint_stem
    metadata = (
        resolved_checkpoint.metadata
        if resolved_checkpoint.metadata is not None
        else load_checkpoint_metadata(checkpoint_stem)
    )
    if metadata is None:
        profile = infer_legacy_profile(checkpoint_stem)
        if scenario_name is not None:
            profile = override_profile_scenario(profile, scenario_name)
        print(
            "No se encontro metadata asociada al checkpoint. "
            "Se usara el escenario del perfil 'default' con el action space inferido del checkpoint."
        )
        return profile.for_evaluation(), checkpoint_stem

    profile = TrainingProfile.from_dict(metadata["profile"]).for_evaluation()
    if scenario_name is not None:
        profile = override_profile_scenario(profile, scenario_name).for_evaluation()
    return profile, checkpoint_stem


def evaluate(
    checkpoint_name: str | None = None,
    steps: int | None = None,
    profile_name: str = DEFAULT_PROFILE_NAME,
    checkpoint_selection: CheckpointSelection = "best",
    scenario_name: str | None = None,
) -> None:
    project_paths = build_project_paths()
    profile, resolved_checkpoint_stem = resolve_profile_for_evaluation(
        checkpoint_name,
        profile_name=profile_name,
        checkpoint_selection=checkpoint_selection,
        scenario_name=scenario_name,
    )
    profile.validate(project_paths)

    env = make_vectorized_env(profile, project_paths)
    env.seed(profile.seed)
    model = RecurrentPPO.load(str(checkpoint_zip_path(resolved_checkpoint_stem)), env=env)

    observation = cast(np.ndarray, env.reset())
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    print(f"Evaluando checkpoint: {checkpoint_zip_path(resolved_checkpoint_stem)}.")
    print(
        f"Escenario de evaluacion: {profile.scenario_name} "
        f"({profile.scenario_key}). Presiona Ctrl+C para salir."
    )
    try:
        step_count = 0
        while steps is None or step_count < steps:
            action, lstm_states = model.predict(
                observation,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            step_result = env.step(action)
            observation = cast(np.ndarray, step_result[0])
            dones = step_result[2]
            episode_starts = dones
            env.render()
            step_count += 1
        print(f"Evaluacion completada: {step_count} pasos ejecutados.")
    except KeyboardInterrupt:
        print("Evaluacion detenida por el usuario.")
    except ViZDoomUnexpectedExitException:
        print("Evaluacion detenida: la ventana de ViZDoom fue cerrada.")
    finally:
        env.close()
