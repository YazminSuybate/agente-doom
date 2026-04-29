from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doom_agent.config import build_project_paths, get_training_profile
from doom_agent.shared.contracts import EvaluationMetricsPayload
from doom_agent.utils.reports import (
    build_run_id,
    build_training_run_report,
    list_experiment_runs,
    report_path_for_run,
    save_training_run_report,
)


class ReportTests(unittest.TestCase):
    def test_save_training_run_report_updates_index(self) -> None:
        root_dir = Path("artifacts") / "test-temp" / "reports"
        shutil.rmtree(root_dir, ignore_errors=True)
        project_paths = build_project_paths(root_dir=root_dir)
        profile = get_training_profile("fast", seed=123)
        run_id = build_run_id(profile)
        report = build_training_run_report(
            run_id=run_id,
            created_at_utc="2026-01-01T00:00:00+00:00",
            profile_name="fast",
            run_label="manual:test",
            profile=profile,
            checkpoint_path=project_paths.checkpoints_dir / f"{profile.checkpoint_name}.zip",
            best_checkpoint_path=project_paths.checkpoints_dir
            / f"{profile.checkpoint_name}_best.zip",
            training_status="completed",
            completed=True,
            saved_timesteps=profile.effective_timesteps,
            resume_mode="auto",
            resume_source=None,
            resume_saved_timesteps=None,
            evaluation_metrics=EvaluationMetricsPayload(
                mean_reward=10.0,
                std_reward=1.0,
                mean_episode_length=30.0,
                episodes=5,
            ),
            duration_seconds=12.5,
            stopped_early=False,
            stop_reason=None,
        )

        try:
            report_path = save_training_run_report(project_paths, report)
            self.assertTrue(report_path.exists())
            self.assertEqual(report_path, report_path_for_run(project_paths, run_id))

            runs = list_experiment_runs(project_paths)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], run_id)
            self.assertEqual(runs[0]["mean_reward"], 10.0)
            self.assertEqual(runs[0]["run_label"], "manual:test")
            self.assertEqual(runs[0]["seed"], 123)
        finally:
            shutil.rmtree(root_dir, ignore_errors=True)
