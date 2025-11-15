import math
from typing import Dict, List, Optional

from dt.cluster_manager import ClusterManager
from dt.policy.base import Policy
from dt.predict import PredictiveSimulator
from dt.state import DTState, Job, JobStage, PlacementDecision, Node


class ResilientPolicy(Policy):
    def __init__(
        self,
        state: DTState,
        simulator: PredictiveSimulator,
        cluster_manager: Optional[ClusterManager] = None,
    ) -> None:
        super().__init__(state, simulator)
        self.cluster_manager = cluster_manager

    def _reliability(self, node: Node) -> float:
        penalty = 0.2 if not node.available else 0.0
        util = max(node.tel.cpu_util, node.tel.mem_util)
        return max(0.0, 1.0 - penalty - 0.5 * util)

    def _candidate_nodes(self, stage: JobStage) -> List[Node]:
        nodes: List[Node] = []
        for node in self.state.list_nodes():
            if (
                stage.compute.gpu_vram_gb > 0
                and node.hardware.gpu_vram_gb < stage.compute.gpu_vram_gb
            ):
                continue
            nodes.append(node)
        return nodes

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
            candidate_node.name,
        )
        return latency_ms

    def place(self, job: Job) -> Dict[str, PlacementDecision]:
        placements: Dict[str, PlacementDecision] = {}
        prev_node_for: Dict[str, Node] = {}
        for stage in job.stages:
            best_node: Optional[Node] = None
            best_score = -math.inf
            best_format = "native"
            for node in self._candidate_nodes(stage):
                exec_format = self.sim.choose_exec_format(stage, node)
                lat = self.sim.compute_stage_latency_ms(stage, node, exec_format)

                if stage.predecessor and stage.predecessor in prev_node_for:
                    lat += self.sim.compute_network_delay_ms(
                        prev_node_for[stage.predecessor], node
                    )

                if not stage.predecessor and job.origin:
                    lat += self._compute_origin_latency(job, node)

                rel = self._reliability(node)
                score = rel - 0.001 * lat
                if score > best_score:
                    best_node = node
                    best_score = score
                    best_format = exec_format

            if best_node is None:
                continue

            placements[stage.id] = PlacementDecision(
                stage_id=stage.id,
                node_name=best_node.name,
                exec_format=best_format,
            )
            prev_node_for[stage.id] = best_node

        return placements
