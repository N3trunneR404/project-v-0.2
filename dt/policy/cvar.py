from __future__ import annotations

from typing import Dict, List
import numpy as np

from dt.state import DTState, Job, JobStage, PlacementDecision, Node
from dt.predict import PredictiveSimulator
from dt.policy.base import Policy
from dt.cluster_manager import ClusterManager
from typing import Optional


class RiskAwareCvarPolicy(Policy):
	def __init__(
		self,
		state: DTState,
		simulator: PredictiveSimulator,
		alpha: float = 0.9,
		cluster_manager: Optional[ClusterManager] = None
	) -> None:
		super().__init__(state, simulator)
		self.alpha = alpha
		self.cluster_manager = cluster_manager

	def _candidate_nodes(self, stage: JobStage) -> List[Node]:
		return self.state.list_nodes()

	def _compute_origin_latency(self, job: Job, candidate_node: Node) -> float:
		"""Compute latency from job origin to candidate node."""
		if not job.origin or not self.cluster_manager:
			return 0.0
		
		candidate_cluster = self.state.get_cluster(candidate_node.name)
		if not candidate_cluster:
			return 0.0
		
		origin_cluster = job.origin.cluster
		latency_ms = self.cluster_manager.get_latency_between(
			origin_cluster,
			candidate_cluster,
			job.origin.node,
			candidate_node.name
		)
		return latency_ms

	def _sample_cost(self, job: Job, placements: Dict[str, PlacementDecision], runs: int = 16) -> float:
		samples = []
		for _ in range(runs):
			noise = np.random.lognormal(mean=0.0, sigma=0.15)
			res = self.sim.score_plan(job, placements)
			
			# Add origin latency for first stage
			origin_lat = 0.0
			if job.origin and job.stages:
				first_stage = job.stages[0]
				if first_stage.id in placements:
					decision = placements[first_stage.id]
					candidate_node = self.state.get_node(decision.node_name)
					if candidate_node:
						origin_lat = self._compute_origin_latency(job, candidate_node)
			
			samples.append((res.latency_ms + origin_lat) * noise)
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





