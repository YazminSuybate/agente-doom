from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from vizdoom import ViZDoomUnexpectedExitException

from doom_agent.config import (
    build_project_paths,
    get_training_profile,
    materialize_curriculum_profiles,
)
from doom_agent.config.schema import TrainingProfile
from doom_agent.envs import make_vectorized_env
from doom_agent.services.training_support import (
    AUTO_RESUME_MODE,
    EvaluationSettings,
    PeriodicTrainingCallback,
    ResumeState,
    load_training_model,
    resolve_resume_state,
)
from doom_agent.utils.checkpoints import (
    best_checkpoint_stem,
    build_checkpoint_metadata,
    checkpoint_zip_path,
    save_checkpoint_bundle,
)
from doom_agent.utils.filesystem import ensure_directories
from doom_agent.utils.reports import (
    build_run_id,
    build_training_run_report,
    save_training_run_report,
)


@dataclass(frozen=True, slots=True)
class TrainingExecutionResult:
    profile_name: str
    profile: TrainingProfile
    final_checkpoint_path: Path
    report_path: Path
    training_status: str
    completed: bool
    saved_timesteps: int
    stopped_early: bool
    stop_reason: str | None


def print_training_summary(
    profile_name: str,
    profile: TrainingProfile,
    checkpoints_dir: Path,
    resume_state: ResumeState,
    evaluation_settings: EvaluationSettings,
    run_label: str | None = None,
) -> None:
    if run_label:
        print(f"Iniciando entrenamiento '{profile_name}' ({run_label}) en {profile.scenario_name}.")
    else:
        print(f"Iniciando entrenamiento '{profile_name}' en {profile.scenario_name}.")
    print(f"Seed: {profile.seed}.")
    if profile.scenario_description:
        print(f"Escenario '{profile.scenario_key}': {profile.scenario_description}")
    print(f"Acciones: {profile.action_space_kind} con {profile.frame_stack} frames apilados.")
    print(
        "Reward shaping: "
        f"scale={profile.reward_shaping.scale}, "
        f"offset={profile.reward_shaping.offset}, "
        f"clip_min={profile.reward_shaping.clip_min}, "
        f"clip_max={profile.reward_shaping.clip_max}."
    )
    print(f"Timesteps solicitados para esta ejecucion: {profile.requested_timesteps}.")
    print(f"Timesteps efectivos de esta ejecucion: {profile.effective_timesteps}.")
    if profile.uses_rounded_timesteps:
        print(
            "Stable-Baselines3 redondea al siguiente multiplo de "
            f"n_steps={profile.n_steps}; por eso se ejecutaran "
            f"{profile.effective_timesteps} pasos."
        )

    if resume_state.is_resumed:
        print(
            "Reanudando desde "
            f"{resume_state.resume_source} con {resume_state.resume_saved_timesteps} pasos acumulados."
        )
        if resume_state.note:
            print(resume_state.note)
    else:
        if resume_state.mode == "from_scratch":
            print("Entrenamiento forzado desde cero.")
        elif resume_state.mode == AUTO_RESUME_MODE:
            print("No se encontro checkpoint compatible. El entrenamiento comienza desde cero.")
        else:
            print("Entrenamiento comenzando desde cero.")

    print(
        f"Checkpoints automaticos cada {profile.checkpoint_frequency} pasos en {checkpoints_dir}."
    )
    print(
        f"Evaluacion periodica cada {evaluation_settings.frequency} pasos "
        f"durante {evaluation_settings.episodes} episodios."
    )
    if profile.early_stopping.enabled:
        print(
            "Early stopping activo: "
            f"patience={profile.early_stopping.patience_evaluations}, "
            f"min_evaluations={profile.early_stopping.min_evaluations}, "
            f"min_delta={profile.early_stopping.min_delta}."
        )


def train_profile(
    profile_name: str,
    profile: TrainingProfile,
    *,
    resume_mode: str = AUTO_RESUME_MODE,
    from_scratch: bool = False,
    eval_frequency: int | None = None,
    eval_episodes: int = 5,
    save_best: bool = True,
    allow_scenario_resume: bool = False,
    run_label: str | None = None,
) -> TrainingExecutionResult:
    project_paths = build_project_paths()
    if eval_episodes <= 0:
        raise ValueError("'eval_episodes' debe ser mayor que cero.")
    if eval_frequency is not None and eval_frequency <= 0:
        raise ValueError("'eval_frequency' debe ser mayor que cero.")

    profile.validate(project_paths)

    ensure_directories(
        [
            project_paths.artifacts_dir,
            project_paths.checkpoints_dir,
            project_paths.auto_checkpoints_dir,
            project_paths.tensorboard_dir,
            project_paths.reports_dir,
        ]
    )

    resume_state = resolve_resume_state(
        project_paths,
        profile,
        resume_mode=resume_mode,
        from_scratch=from_scratch,
        allow_scenario_change=allow_scenario_resume,
    )
    evaluation_settings = EvaluationSettings(
        frequency=eval_frequency or profile.checkpoint_frequency,
        episodes=eval_episodes,
        save_best=save_best,
        best_checkpoint_stem=project_paths.checkpoints_dir / f"{profile.checkpoint_name}_best",
    )

    env = make_vectorized_env(profile, project_paths)
    env.seed(profile.seed)
    eval_profile = profile.for_evaluation(render=False).with_seed(profile.seed + 1)
    eval_env = make_vectorized_env(eval_profile, project_paths)
    eval_env.seed(eval_profile.seed)
    model = load_training_model(env, profile, project_paths.tensorboard_dir, resume_state)
    callback = PeriodicTrainingCallback(
        profile_name=profile_name,
        profile=profile,
        auto_checkpoint_dir=project_paths.auto_checkpoints_dir,
        evaluation_settings=evaluation_settings,
        eval_env=eval_env,
        resume_state=resume_state,
        verbose=1,
    )

    final_checkpoint_stem = project_paths.checkpoints_dir / profile.checkpoint_name
    run_created_at = datetime.now(UTC)
    run_id = build_run_id(profile, run_created_at)
    started_at = perf_counter()
    final_checkpoint_path = checkpoint_zip_path(final_checkpoint_stem)
    report_path: Path | None = None
    print_training_summary(
        profile_name,
        profile,
        project_paths.auto_checkpoints_dir,
        resume_state,
        evaluation_settings,
        run_label=run_label,
    )

    completed = False
    training_status = "interrupted"
    try:
        model.learn(
            total_timesteps=profile.effective_timesteps,
            tb_log_name=profile.tensorboard_run_name,
            callback=callback,
            reset_num_timesteps=not resume_state.is_resumed,
        )
        completed = True
        training_status = "completed"
        if callback.early_stopping.stopped:
            training_status = "early_stopped"
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido por el usuario. Guardando progreso...")
        training_status = "keyboard_interrupt"
    except ViZDoomUnexpectedExitException:
        print("Entrenamiento detenido: ViZDoom se cerro. Guardando progreso...")
        training_status = "vizdoom_exit"
    finally:
        duration_seconds = perf_counter() - started_at
        metadata = build_checkpoint_metadata(
            profile_name=profile_name,
            profile=profile,
            saved_timesteps=model.num_timesteps,
            resume_source=resume_state.resume_source,
            resume_saved_timesteps=resume_state.resume_saved_timesteps,
            training_status=training_status,
            evaluation_metrics=callback.last_evaluation_metrics,
            is_best_checkpoint=False,
        )
        save_checkpoint_bundle(model, final_checkpoint_stem, metadata)
        try:
            env.close()
        except ViZDoomUnexpectedExitException:
            pass
        try:
            eval_env.close()
        except ViZDoomUnexpectedExitException:
            pass

        best_checkpoint_path = checkpoint_zip_path(best_checkpoint_stem(final_checkpoint_stem))
        report = build_training_run_report(
            run_id=run_id,
            created_at_utc=run_created_at.isoformat(),
            profile_name=profile_name,
            run_label=run_label,
            profile=profile,
            checkpoint_path=final_checkpoint_path,
            best_checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else None,
            training_status=training_status,
            completed=completed,
            saved_timesteps=model.num_timesteps,
            resume_mode=resume_state.mode,
            resume_source=resume_state.resume_source,
            resume_saved_timesteps=resume_state.resume_saved_timesteps,
            evaluation_metrics=callback.last_evaluation_metrics,
            duration_seconds=duration_seconds,
            stopped_early=callback.early_stopping.stopped,
            stop_reason=callback.early_stopping.stop_reason,
        )
        report_path = save_training_run_report(project_paths, report)

        if training_status == "early_stopped":
            print(f"Entrenamiento detenido anticipadamente. Modelo guardado en {final_checkpoint_stem.with_suffix('.zip')}")
            if callback.early_stopping.stop_reason is not None:
                print(callback.early_stopping.stop_reason)
        elif completed:
            print(
                f"Entrenamiento completado. Modelo guardado en {final_checkpoint_stem.with_suffix('.zip')}"
            )
        else:
            print(f"Progreso guardado en {final_checkpoint_stem.with_suffix('.zip')}")
        print(f"Reporte de entrenamiento guardado en {report_path}")
    if report_path is None:
        raise RuntimeError("No se pudo guardar el reporte de entrenamiento.")
    return TrainingExecutionResult(
        profile_name=profile_name,
        profile=profile,
        final_checkpoint_path=final_checkpoint_path,
        report_path=report_path,
        training_status=training_status,
        completed=completed,
        saved_timesteps=model.num_timesteps,
        stopped_early=callback.early_stopping.stopped,
        stop_reason=callback.early_stopping.stop_reason,
    )


def train(
    profile_name: str = "default",
    requested_timesteps: int | None = None,
    scenario_name: str | None = None,
    seed: int | None = None,
    *,
    resume_mode: str = AUTO_RESUME_MODE,
    from_scratch: bool = False,
    eval_frequency: int | None = None,
    eval_episodes: int = 5,
    save_best: bool = True,
) -> TrainingExecutionResult:
    profile = get_training_profile(
        profile_name,
        requested_timesteps=requested_timesteps,
        scenario_name=scenario_name,
        seed=seed,
    )
    if profile.curriculum:
        if scenario_name is not None:
            raise ValueError(
                "No puedes usar '--scenario' con un perfil de curriculum; las etapas ya definen sus escenarios."
            )
        stages = materialize_curriculum_profiles(
            profile_name=profile_name,
            requested_timesteps=requested_timesteps,
            seed=seed,
        )
        print(f"Ejecutando curriculum de {len(stages)} etapas.")
        stage_resume_mode = resume_mode
        stage_from_scratch = from_scratch
        last_result: TrainingExecutionResult | None = None

        for stage_index, stage_profile in enumerate(stages, start=1):
            last_result = train_profile(
                profile_name=profile_name,
                profile=stage_profile,
                resume_mode=stage_resume_mode,
                from_scratch=stage_from_scratch,
                eval_frequency=eval_frequency,
                eval_episodes=eval_episodes,
                save_best=save_best,
                allow_scenario_resume=stage_index > 1,
                run_label=f"curriculum:{stage_index}/{len(stages)}",
            )
            if last_result.training_status in {"keyboard_interrupt", "vizdoom_exit"}:
                break
            stage_resume_mode = str(last_result.final_checkpoint_path)
            stage_from_scratch = False

        if last_result is None:
            raise RuntimeError("No se pudo ejecutar ninguna etapa del curriculum.")
        return last_result

    return train_profile(
        profile_name=profile_name,
        profile=profile,
        resume_mode=resume_mode,
        from_scratch=from_scratch,
        eval_frequency=eval_frequency,
        eval_episodes=eval_episodes,
        save_best=save_best,
    )
