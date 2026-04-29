from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import cast

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doom_agent.config import build_project_paths, get_training_profile
from doom_agent.envs import make_vectorized_env


class EnvironmentSmokeTests(unittest.TestCase):
    def test_environment_reset_and_step_support_combined_actions(self) -> None:
        project_paths = build_project_paths()
        profile = get_training_profile("fast")
        env = make_vectorized_env(profile, project_paths)

        try:
            observation = cast(np.ndarray, env.reset())
            self.assertEqual(
                observation.shape,
                (1, profile.frame_stack, profile.screen_height, profile.screen_width),
            )

            step_result = env.step(np.array([[1, 0, 1]], dtype=np.int64))
            observation = cast(np.ndarray, step_result[0])
            rewards, dones = step_result[1], step_result[2]
            self.assertEqual(
                observation.shape,
                (1, profile.frame_stack, profile.screen_height, profile.screen_width),
            )
            self.assertEqual(rewards.shape, (1,))
            self.assertEqual(dones.shape, (1,))
        finally:
            env.close()
