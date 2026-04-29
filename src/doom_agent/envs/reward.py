from __future__ import annotations

from doom_agent.config.schema import RewardShapingConfig


class RewardShaper:
    """Applies deterministic reward shaping configured outside the environment code."""

    def __init__(self, config: RewardShapingConfig) -> None:
        self.config = config

    def apply(self, reward: float) -> float:
        return self.config.apply(reward)
