from __future__ import annotations

import argparse
import json

from doom_agent.cli.evaluate import add_evaluate_arguments
from doom_agent.cli.train import add_train_arguments
from doom_agent.config import (
    DEFAULT_PROFILE_NAME,
    PROFILE_NAMES,
    SCENARIO_NAMES,
    build_project_paths,
    get_training_profile,
)
from doom_agent.services.evaluator import evaluate
from doom_agent.services.sweeps import run_sweep
from doom_agent.services.trainer import train
from doom_agent.utils.checkpoints import list_all_checkpoints, resolve_checkpoint_preference
from doom_agent.utils.reports import list_experiment_runs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI unificada para el proyecto Agente Doom.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena el agente.")
    add_train_arguments(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evalua un checkpoint.")
    add_evaluate_arguments(evaluate_parser)

    sweep_parser = subparsers.add_parser("sweep", help="Ejecuta un sweep secuencial.")
    sweep_parser.add_argument(
        "--config",
        choices=PROFILE_NAMES,
        default="fast",
        help="Perfil base para el sweep.",
    )
    sweep_parser.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        default=None,
        help="Escenario base para el sweep.",
    )
    sweep_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Sobrescribe los timesteps solicitados para todas las variantes.",
    )
    sweep_parser.add_argument(
        "--learning-rates",
        default=None,
        help="Lista CSV de learning rates. Ejemplo: 0.0001,0.0003",
    )
    sweep_parser.add_argument(
        "--n-steps-values",
        default=None,
        help="Lista CSV de n_steps. Ejemplo: 256,512",
    )
    sweep_parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Lista CSV de batch sizes. Ejemplo: 32,64",
    )
    sweep_parser.add_argument(
        "--seeds",
        default=None,
        help="Lista CSV de seeds. Ejemplo: 42,43",
    )
    sweep_parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Frecuencia de evaluacion periodica en pasos.",
    )
    sweep_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Cantidad de episodios por evaluacion periodica.",
    )
    sweep_parser.add_argument(
        "--no-save-best",
        action="store_true",
        help="No guardar el mejor modelo por variante.",
    )

    subparsers.add_parser("list-profiles", help="Lista perfiles y escenarios disponibles.")

    list_checkpoints_parser = subparsers.add_parser(
        "list-checkpoints", help="Lista checkpoints conocidos."
    )
    list_checkpoints_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Cantidad maxima de checkpoints a mostrar.",
    )

    list_runs_parser = subparsers.add_parser(
        "list-runs", help="Lista reportes de entrenamiento registrados."
    )
    list_runs_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Cantidad maxima de corridas a mostrar.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect-checkpoint", help="Muestra la metadata de un checkpoint."
    )
    add_evaluate_arguments(inspect_parser)
    return parser


def _print_profiles() -> None:
    print("Perfiles:")
    for profile_name in PROFILE_NAMES:
        profile = get_training_profile(profile_name)
        print(
            f"- {profile_name}: scenario={profile.scenario_key}, "
            f"timesteps={profile.requested_timesteps}, seed={profile.seed}, "
            f"checkpoint={profile.checkpoint_name}, curriculum={len(profile.curriculum)} etapas"
        )

    print("Escenarios:")
    for scenario_name in SCENARIO_NAMES:
        profile = get_training_profile(DEFAULT_PROFILE_NAME, scenario_name=scenario_name)
        print(
            f"- {scenario_name}: file={profile.scenario_name}, "
            f"checkpoint={profile.checkpoint_name}"
        )


def _print_checkpoints(limit: int) -> None:
    checkpoints = list_all_checkpoints(build_project_paths())[:limit]
    if not checkpoints:
        print("No se encontraron checkpoints.")
        return

    for checkpoint in checkpoints:
        metadata = checkpoint.metadata
        training_status = "legacy" if metadata is None else metadata["training_status"]
        profile_name = "legacy" if metadata is None else metadata["profile_name"]
        mean_reward = None
        if metadata is not None and metadata["evaluation_metrics"] is not None:
            mean_reward = metadata["evaluation_metrics"]["mean_reward"]
        print(
            f"- {checkpoint.checkpoint_stem.with_suffix('.zip')}: "
            f"profile={profile_name}, timesteps={checkpoint.saved_timesteps}, "
            f"status={training_status}, mean_reward={mean_reward}"
        )


def _print_runs(limit: int) -> None:
    runs = list_experiment_runs(build_project_paths())[:limit]
    if not runs:
        print("No se encontraron reportes de entrenamiento.")
        return

    for run in runs:
        print(
            f"- {run['created_at_utc']}: run_id={run['run_id']}, "
            f"profile={run['profile_name']}, scenario={run['scenario_key']}, "
            f"timesteps={run['saved_timesteps']}, status={run['training_status']}, "
            f"mean_reward={run['mean_reward']}, report={run['report_path']}"
        )


def _inspect_checkpoint(args: argparse.Namespace) -> None:
    project_paths = build_project_paths()
    checkpoint_name = args.checkpoint
    if checkpoint_name is None:
        checkpoint_name = get_training_profile(
            args.config,
            scenario_name=args.scenario,
        ).checkpoint_name

    checkpoint = resolve_checkpoint_preference(
        project_paths,
        checkpoint_name,
        preference=args.select,
    )
    print(f"Checkpoint resuelto: {checkpoint.checkpoint_stem.with_suffix('.zip')}")
    print(f"Timesteps guardados: {checkpoint.saved_timesteps}")
    if checkpoint.metadata is None:
        print("Checkpoint legacy sin metadata estructurada.")
        return

    print(json.dumps(checkpoint.metadata, indent=2, sort_keys=True))


def _parse_float_csv(values: str | None, fallback: float) -> tuple[float, ...]:
    if values is None:
        return (fallback,)
    return tuple(float(value.strip()) for value in values.split(",") if value.strip())


def _parse_int_csv(values: str | None, fallback: int) -> tuple[int, ...]:
    if values is None:
        return (fallback,)
    return tuple(int(value.strip()) for value in values.split(",") if value.strip())


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        if args.from_scratch and args.resume not in {"auto", "latest"}:
            raise SystemExit(
                "No puedes usar '--from-scratch' junto con un checkpoint explicito en '--resume'."
            )
        train(
            profile_name=args.config,
            requested_timesteps=args.timesteps,
            scenario_name=args.scenario,
            seed=args.seed,
            resume_mode=args.resume,
            from_scratch=args.from_scratch,
            eval_frequency=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_best=not args.no_save_best,
        )
        return

    if args.command == "evaluate":
        evaluate(
            checkpoint_name=args.checkpoint,
            steps=args.steps,
            profile_name=args.config,
            checkpoint_selection=args.select,
            scenario_name=args.scenario,
        )
        return

    if args.command == "sweep":
        base_profile = get_training_profile(
            profile_name=args.config,
            requested_timesteps=args.timesteps,
            scenario_name=args.scenario,
        )
        run_sweep(
            profile_name=args.config,
            scenario_name=args.scenario,
            requested_timesteps=args.timesteps,
            learning_rates=_parse_float_csv(args.learning_rates, base_profile.learning_rate),
            n_steps_values=_parse_int_csv(args.n_steps_values, base_profile.n_steps),
            batch_sizes=_parse_int_csv(args.batch_sizes, base_profile.batch_size),
            seeds=_parse_int_csv(args.seeds, base_profile.seed),
            eval_frequency=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_best=not args.no_save_best,
        )
        return

    if args.command == "list-profiles":
        _print_profiles()
        return

    if args.command == "list-checkpoints":
        _print_checkpoints(args.limit)
        return

    if args.command == "list-runs":
        _print_runs(args.limit)
        return

    if args.command == "inspect-checkpoint":
        _inspect_checkpoint(args)
        return

    raise SystemExit(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
