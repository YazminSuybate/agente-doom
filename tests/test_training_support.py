from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doom_agent.config.schema import EarlyStoppingConfig
from doom_agent.services.training_support import EarlyStoppingTracker


class EarlyStoppingTests(unittest.TestCase):
    def test_early_stopping_tracker_stops_after_patience(self) -> None:
        tracker = EarlyStoppingTracker(
            config=EarlyStoppingConfig(
                enabled=True,
                patience_evaluations=2,
                min_evaluations=2,
                min_delta=0.1,
            )
        )

        self.assertTrue(tracker.register(1.0))
        self.assertFalse(tracker.stopped)

        self.assertFalse(tracker.register(1.05))
        self.assertFalse(tracker.stopped)

        self.assertFalse(tracker.register(1.08))
        self.assertTrue(tracker.stopped)
        self.assertIsNotNone(tracker.stop_reason)
