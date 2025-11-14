"""
Placeholder RL policy interface.

The current release focuses on heuristic schedulers (greedy/resilient/CVaR).
This stub keeps the import surface stable so future RL integrations can drop
in without touching callers.
"""

from __future__ import annotations

from typing import Dict, Any


class RLPolicy:
	def __init__(self, *args, **kwargs) -> None:
		self.enabled = False

	def choose_node(self, stage: Dict[str, Any], candidates: Dict[str, Any]) -> str:
		raise NotImplementedError("RLPolicy is not implemented in this release.")

	def record_transition(self, *args, **kwargs) -> None:
		raise NotImplementedError("RLPolicy is not implemented in this release.")

