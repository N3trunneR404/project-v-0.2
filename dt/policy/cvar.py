from __future__ import annotations

from typing import Dict, List
import numpy as np

from dt.state import DTState, Job, JobStage, PlacementDecision, Node
from dt.predict import PredictiveSimulator
from dt.policy.base import Policy


class RiskAwareCvarPolicy(Policy):
	def __init__(self, state: DTState, simulator: PredictiveSimulator, alpha: float = 0.9) -> None:
		super().__init__(state, simulator)
		self.alpha = alpha

	def _candidate_nodes(self, stage: JobStage) -> List[Node]:
		return self.state.list_nodes()

	def _sample_cost(self, job: Job, placements: Dict[str, PlacementDecision], runs: int = 16) -> float:
		samples = []
		for _ in range(runs):
			noise = np.random.lognormal(mean=0.0, sigma=0.15)
			res = self.sim.score_plan(job, placements)
			samples.append(res.latency_ms * noise)
		q = np.quantile(samples, self.alpha)
		tail = [s for s in samples if s >= q]
		return float(np.mean(tail)) if tail else float(q)

	def place(self, job: Job) -> Dict[str, PlacementDecision]:
		placements: Dict[str, PlacementDecision] = {}
		for stage in job.stages:
			best_dec: PlacementDecision | None = None
			best_cvar = float("inf")
			for node in self._candidate_nodes(stage):
				exec_format = self.sim.choose_exec_format(stage, node)
				test = dict(placements)
				test[stage.id] = PlacementDecision(stage_id=stage.id, node_name=node.name, exec_format=exec_format)
				cvar = self._sample_cost(job, test)
				if cvar < best_cvar:
					best_cvar = cvar
					best_dec = test[stage.id]
			if best_dec:
				placements[stage.id] = best_dec
		return placements





