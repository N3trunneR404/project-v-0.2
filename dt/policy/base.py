from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
from dt.state import Job, PlacementDecision, DTState
from dt.predict import PredictiveSimulator, SimulationResult


class Policy(ABC):
	def __init__(self, state: DTState, simulator: PredictiveSimulator) -> None:
		self.state = state
		self.sim = simulator

	@abstractmethod
	def place(self, job: Job) -> Dict[str, PlacementDecision]:
		raise NotImplementedError

	def score(self, job: Job, placements: Dict[str, PlacementDecision]) -> SimulationResult:
		return self.sim.score_plan(job, placements)





