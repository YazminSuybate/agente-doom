from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doom_agent.config import (
    PROFILE_NAMES,
    SCENARIO_NAMES,
    build_project_paths,
    get_training_profile,
    load_training_catalog,
    materialize_curriculum_profiles,
)
from doom_agent.config.schema import RewardShapingConfig


class TrainingProfileTests(unittest.TestCase):
    def test_training_catalog_is_loaded_from_toml(self) -> None:
        catalog = load_training_catalog()
        self.assertIn("default", catalog.profiles)
        self.assertIn("deadly_corridor", catalog.scenarios)

    def test_requested_timesteps_are_rounded_explicitly(self) -> None:
        profile = get_training_profile("fast", requested_timesteps=8)
        self.assertEqual(profile.requested_timesteps, 8)
        self.assertEqual(profile.effective_timesteps, 256)
        self.assertTrue(profile.uses_rounded_timesteps)

    def test_profile_can_be_materialized_for_another_scenario(self) -> None:
        profile = get_training_profile("fast", scenario_name="deadly_corridor")
        self.assertEqual(profile.scenario_key, "deadly_corridor")
        self.assertEqual(profile.scenario_name, "deadly_corridor.cfg")
        self.assertIn("__deadly_corridor", profile.checkpoint_name)
        self.assertEqual(profile.reward_shaping.clip_min, -1.0)
        self.assertEqual(profile.reward_shaping.clip_max, 1.0)

    def test_reward_shaping_applies_scale_offset_and_clip(self) -> None:
        shaping = RewardShapingConfig(scale=0.5, offset=1.0, clip_min=-2.0, clip_max=3.0)
        self.assertEqual(shaping.apply(2.0), 1.5)
        self.assertEqual(shaping.apply(10.0), 3.0)
        self.assertEqual(shaping.apply(-10.0), -2.0)

    def test_profile_supports_seed_default_and_override(self) -> None:
        default_profile = get_training_profile("fast")
        custom_profile = get_training_profile("fast", seed=123)
        self.assertEqual(default_profile.seed, 42)
        self.assertEqual(custom_profile.seed, 123)

    def test_curriculum_profiles_are_materialized_in_stage_order(self) -> None:
        stages = materialize_curriculum_profiles("curriculum_fast", seed=100)
        self.assertEqual(len(stages), 3)
        self.assertEqual([stage.scenario_key for stage in stages], ["basic", "defend_the_center", "health_gathering"])
        self.assertEqual([stage.seed for stage in stages], [100, 101, 102])
        self.assertTrue(all(stage.requested_timesteps == 2048 for stage in stages))

    def test_profiles_are_valid_against_existing_scenarios(self) -> None:
        project_paths = build_project_paths()
        for profile_name in PROFILE_NAMES:
            with self.subTest(profile=profile_name):
                get_training_profile(profile_name).validate(project_paths)

        for scenario_name in SCENARIO_NAMES:
            with self.subTest(scenario=scenario_name):
                get_training_profile("fast", scenario_name=scenario_name).validate(project_paths)
