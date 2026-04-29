from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from doom_agent.config.schema import ProjectPaths, TrainingProfile
from doom_agent.shared.contracts import (
    EvaluationMetricsPayload,
    ExperimentIndexEntryPayload,
    ExperimentIndexPayload,
    TrainingRunReportPayload,
)
from doom_agent.utils.filesystem import ensure_directories, read_json, write_json


def build_run_id(profile: TrainingProfile, created_at_utc: datetime | None = None) -> str:
    timestamp = (created_at_utc or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{profile.checkpoint_name}__{timestamp}"


def build_training_run_report(
    *,
    run_id: str,
    created_at_utc: str,
    profile_name: str,
    run_label: str | None,
    profile: TrainingProfile,
    checkpoint_path: Path,
    best_checkpoint_path: Path | None,
    training_status: str,
    completed: bool,
    saved_timesteps: int,
    resume_mode: str,
    resume_source: str | None,
    resume_saved_timesteps: int | None,
    evaluation_metrics: EvaluationMetricsPayload | None,
    duration_seconds: float,
    stopped_early: bool,
    stop_reason: str | None,
) -> TrainingRunReportPayload:
    return {
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "profile_name": profile_name,
        "run_label": run_label,
        "scenario_key": profile.scenario_key,
        "scenario_name": profile.scenario_name,
        "checkpoint_name": profile.checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path)
        if best_checkpoint_path is not None
        else None,
        "training_status": training_status,
        "completed": completed,
        "requested_timesteps": profile.requested_timesteps,
        "effective_timesteps": profile.effective_timesteps,
        "saved_timesteps": saved_timesteps,
        "seed": profile.seed,
        "resume_mode": resume_mode,
        "resume_source": resume_source,
        "resume_saved_timesteps": resume_saved_timesteps,
        "evaluation_metrics": evaluation_metrics,
        "duration_seconds": duration_seconds,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }


def report_path_for_run(project_paths: ProjectPaths, run_id: str) -> Path:
    return project_paths.reports_dir / f"{run_id}.json"


def build_experiment_index_entry(
    report: TrainingRunReportPayload,
    report_path: Path,
) -> ExperimentIndexEntryPayload:
    evaluation_metrics = report["evaluation_metrics"]
    mean_reward = None if evaluation_metrics is None else evaluation_metrics["mean_reward"]
    return {
        "run_id": report["run_id"],
        "created_at_utc": report["created_at_utc"],
        "profile_name": report["profile_name"],
        "run_label": report["run_label"],
        "scenario_key": report["scenario_key"],
        "checkpoint_name": report["checkpoint_name"],
        "checkpoint_path": report["checkpoint_path"],
        "best_checkpoint_path": report["best_checkpoint_path"],
        "saved_timesteps": report["saved_timesteps"],
        "training_status": report["training_status"],
        "mean_reward": mean_reward,
        "seed": report["seed"],
        "report_path": str(report_path),
    }


def load_experiment_index(project_paths: ProjectPaths) -> ExperimentIndexPayload:
    index_path = project_paths.experiments_index_path
    if not index_path.exists():
        return {"runs": []}
    return cast(ExperimentIndexPayload, read_json(index_path))


def save_training_run_report(
    project_paths: ProjectPaths,
    report: TrainingRunReportPayload,
) -> Path:
    ensure_directories([project_paths.reports_dir])
    report_path = report_path_for_run(project_paths, report["run_id"])
    write_json(report_path, report)

    index = load_experiment_index(project_paths)
    filtered_runs = [entry for entry in index["runs"] if entry["run_id"] != report["run_id"]]
    filtered_runs.append(build_experiment_index_entry(report, report_path))
    filtered_runs.sort(key=lambda entry: entry["created_at_utc"], reverse=True)
    write_json(project_paths.experiments_index_path, {"runs": filtered_runs})
    return report_path


def list_experiment_runs(project_paths: ProjectPaths) -> list[ExperimentIndexEntryPayload]:
    return load_experiment_index(project_paths)["runs"]
