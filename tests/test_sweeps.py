from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doom_agent.config import get_training_profile
from doom_agent.services.sweeps import build_sweep_run_specs


class SweepTests(unittest.TestCase):
    def test_build_sweep_run_specs_creates_unique_variants(self) -> None:
        profile = get_training_profile("fast")
        specs = build_sweep_run_specs(
            profile,
            learning_rates=(0.0001, 0.0002),
            n_steps_values=(128,),
            batch_sizes=(32, 64),
            seeds=(11,),
        )

        self.assertEqual(len(specs), 4)
        self.assertTrue(specs[0].profile.checkpoint_name.endswith("__sweep_001"))
        self.assertTrue(specs[-1].profile.checkpoint_name.endswith("__sweep_004"))
        self.assertEqual(specs[1].profile.learning_rate, 0.0001)
        self.assertEqual(specs[2].profile.learning_rate, 0.0002)
