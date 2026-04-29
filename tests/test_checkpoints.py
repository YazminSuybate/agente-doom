from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doom_agent.config import build_project_paths, get_training_profile
from doom_agent.services.training_support import resolve_resume_state
from doom_agent.shared.contracts import CheckpointMetadataPayload, EvaluationMetricsPayload
from doom_agent.utils.checkpoints import (
    build_checkpoint_metadata,
    checkpoint_metadata_path,
    checkpoint_zip_path,
    load_checkpoint_metadata,
    resolve_checkpoint_preference,
    resolve_latest_checkpoint,
    save_checkpoint_bundle,
)


class DummyModel:
    def save(self, checkpoint_stem: str) -> None:
        checkpoint_zip_path(Path(checkpoint_stem)).write_text("dummy model", encoding="utf-8")


class CheckpointTests(unittest.TestCase):
    def test_save_checkpoint_bundle_creates_zip_and_metadata(self) -> None:
        profile = get_training_profile("fast", requested_timesteps=8)
        metadata = build_checkpoint_metadata(
            "fast",
            profile,
            saved_timesteps=profile.effective_timesteps,
            resume_source="artifacts/checkpoints/previous.zip",
            resume_saved_timesteps=10240,
            training_status="completed",
            evaluation_metrics=EvaluationMetricsPayload(
                mean_reward=12.5,
                std_reward=0.5,
                mean_episode_length=42.0,
                episodes=3,
            ),
        )

        temp_dir = Path("artifacts") / "test-temp" / "checkpoint-bundle"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_stem = temp_dir / "ppo_doom_recurrent_fast"
            save_checkpoint_bundle(DummyModel(), checkpoint_stem, metadata)

            self.assertTrue(checkpoint_zip_path(checkpoint_stem).exists())
            self.assertTrue(checkpoint_metadata_path(checkpoint_stem).exists())

            loaded_metadata = load_checkpoint_metadata(checkpoint_stem)
            self.assertIsNotNone(loaded_metadata)
            loaded_metadata = cast(CheckpointMetadataPayload, loaded_metadata)
            self.assertEqual(loaded_metadata["profile_name"], "fast")
            self.assertEqual(loaded_metadata["saved_timesteps"], 256)
            self.assertEqual(loaded_metadata["resume_saved_timesteps"], 10240)
            self.assertEqual(loaded_metadata["training_status"], "completed")
            evaluation_metrics = cast(
                EvaluationMetricsPayload, loaded_metadata["evaluation_metrics"]
            )
            self.assertEqual(evaluation_metrics["mean_reward"], 12.5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resolve_latest_checkpoint_prefers_highest_timestep(self) -> None:
        root_dir = Path("artifacts") / "test-temp" / "resume-selection"
        shutil.rmtree(root_dir, ignore_errors=True)
        project_paths = build_project_paths(root_dir=root_dir)
        project_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        project_paths.auto_checkpoints_dir.mkdir(parents=True, exist_ok=True)

        profile = get_training_profile("fast")
        final_checkpoint = project_paths.checkpoints_dir / profile.checkpoint_name
        auto_checkpoint = (
            project_paths.auto_checkpoints_dir / f"{profile.checkpoint_name}_12000_steps"
        )

        try:
            save_checkpoint_bundle(
                DummyModel(),
                final_checkpoint,
                build_checkpoint_metadata("fast", profile, saved_timesteps=10000),
            )
            save_checkpoint_bundle(
                DummyModel(),
                auto_checkpoint,
                build_checkpoint_metadata("fast", profile, saved_timesteps=12000),
            )

            resolved = resolve_latest_checkpoint(project_paths, profile.checkpoint_name)
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(resolved.checkpoint_stem, auto_checkpoint)
            self.assertEqual(resolved.saved_timesteps, 12000)
        finally:
            shutil.rmtree(root_dir, ignore_errors=True)

    def test_resolve_resume_state_rejects_incompatible_checkpoint_metadata(self) -> None:
        root_dir = Path("artifacts") / "test-temp" / "resume-compatibility"
        shutil.rmtree(root_dir, ignore_errors=True)
        project_paths = build_project_paths(root_dir=root_dir)
        project_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        previous_profile = get_training_profile("fast", scenario_name="basic")
        current_profile = get_training_profile("fast", scenario_name="deadly_corridor")
        checkpoint_stem = project_paths.checkpoints_dir / current_profile.checkpoint_name

        try:
            save_checkpoint_bundle(
                DummyModel(),
                checkpoint_stem,
                build_checkpoint_metadata("fast", previous_profile, saved_timesteps=10000),
            )

            with self.assertRaises(ValueError):
                resolve_resume_state(
                    project_paths, current_profile, resume_mode=str(checkpoint_stem)
                )
        finally:
            shutil.rmtree(root_dir, ignore_errors=True)

    def test_resolve_checkpoint_preference_prefers_best_checkpoint(self) -> None:
        root_dir = Path("artifacts") / "test-temp" / "best-selection"
        shutil.rmtree(root_dir, ignore_errors=True)
        project_paths = build_project_paths(root_dir=root_dir)
        project_paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        profile = get_training_profile("fast")
        final_checkpoint = project_paths.checkpoints_dir / profile.checkpoint_name
        best_checkpoint = project_paths.checkpoints_dir / f"{profile.checkpoint_name}_best"

        try:
            save_checkpoint_bundle(
                DummyModel(),
                final_checkpoint,
                build_checkpoint_metadata("fast", profile, saved_timesteps=10000),
            )
            save_checkpoint_bundle(
                DummyModel(),
                best_checkpoint,
                build_checkpoint_metadata(
                    "fast",
                    profile,
                    saved_timesteps=9000,
                    training_status="best_model",
                    is_best_checkpoint=True,
                ),
            )

            resolved = resolve_checkpoint_preference(
                project_paths,
                profile.checkpoint_name,
                preference="best",
            )
            self.assertEqual(resolved.checkpoint_stem, best_checkpoint)
        finally:
            shutil.rmtree(root_dir, ignore_errors=True)
