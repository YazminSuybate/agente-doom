from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

from doom_agent.config import get_training_profile
from doom_agent.config.schema import TrainingProfile
from doom_agent.services.trainer import TrainingExecutionResult, train_profile


@dataclass(frozen=True, slots=True)
class SweepRunSpec:
    index: int
    total: int
    label: str
    profile: TrainingProfile


def _name_suffix(index: int) -> str:
    return f"__sweep_{index:03d}"


def build_sweep_run_specs(
    base_profile: TrainingProfile,
    *,
    learning_rates: tuple[float, ...],
    n_steps_values: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    seeds: tuple[int, ...],
) -> list[SweepRunSpec]:
    if base_profile.curriculum:
        raise ValueError("Los sweeps secuenciales no soportan perfiles con curriculum por ahora.")

    combinations = list(product(learning_rates, n_steps_values, batch_sizes, seeds))
    specs: list[SweepRunSpec] = []
    total = len(combinations)

    for index, (learning_rate, n_steps, batch_size, seed) in enumerate(combinations, start=1):
        suffix = _name_suffix(index)
        profile = replace(
            base_profile,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            seed=seed,
        ).with_names(
            checkpoint_name=f"{base_profile.checkpoint_name}{suffix}",
            tensorboard_run_name=f"{base_profile.tensorboard_run_name}{suffix}",
        )
        label = (
            f"sweep:{index}/{total} "
            f"lr={learning_rate} n_steps={n_steps} batch_size={batch_size} seed={seed}"
        )
        specs.append(
            SweepRunSpec(
                index=index,
                total=total,
                label=label,
                profile=profile,
            )
        )
    return specs


def run_sweep(
    *,
    profile_name: str,
    scenario_name: str | None = None,
    requested_timesteps: int | None = None,
    learning_rates: tuple[float, ...],
    n_steps_values: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    seeds: tuple[int, ...],
    eval_frequency: int | None = None,
    eval_episodes: int = 5,
    save_best: bool = True,
) -> list[TrainingExecutionResult]:
    base_profile = get_training_profile(
        profile_name=profile_name,
        requested_timesteps=requested_timesteps,
        scenario_name=scenario_name,
    )
    specs = build_sweep_run_specs(
        base_profile,
        learning_rates=learning_rates,
        n_steps_values=n_steps_values,
        batch_sizes=batch_sizes,
        seeds=seeds,
    )
    print(f"Ejecutando sweep secuencial de {len(specs)} corridas.")

    results: list[TrainingExecutionResult] = []
    for spec in specs:
        result = train_profile(
            profile_name=profile_name,
            profile=spec.profile,
            resume_mode="auto",
            from_scratch=True,
            eval_frequency=eval_frequency,
            eval_episodes=eval_episodes,
            save_best=save_best,
            run_label=spec.label,
        )
        results.append(result)
    return results
