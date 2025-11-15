# from __future__ import annotations  # Already at top

from typing import Dict, List, Tuple
import math

from dt.state import DTState, Job, JobStage, PlacementDecision, Node
from dt.predict import PredictiveSimulator
from dt.policy.base import Policy
from dt.cluster_manager import ClusterManager
from typing import Optional


class ResilientPolicy(Policy):
	def __init__(
		self,
		state: DTState,
		simulator: PredictiveSimulator,
		cluster_manager: Optional[ClusterManager] = None
	) -> None:
		super().__init__(state, simulator)
		self.cluster_manager = cluster_manager

	def _reliability(self, node: Node) -> float:
		penalty = 0.2 if not node.available else 0.0
		util = max(node.tel.cpu_util, node.tel.mem_util)
		return max(0.0, 1.0 - penalty - 0.5 * util)

	def _candidate_nodes(self, stage: JobStage) -> List[Node]:
		nodes = []
		for node in self.state.list_nodes():
			if stage.compute.gpu_vram_gb > 0 and node.hardware.gpu_vram_gb < stage.compute.gpu_vram_gb:
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
			candidate_node.name
		)
		return latency_ms

	def place(self, job: Job) -> Dict[str, PlacementDecision]:
		placements: Dict[str, PlacementDecision] = {}
		prev_node_for: Dict[str, Node] = {}
		for stage in job.stages:
			best = (None, -math.inf, "native")  # node, score, format
			for node in self._candidate_nodes(stage):
				exec_format = self.sim.choose_exec_format(stage, node)
				lat = self.sim.compute_stage_latency_ms(stage, node, exec_format)
				
				# Add network delay from predecessor
				if stage.predecessor and stage.predecessor in prev_node_for:
					lat += self.sim.compute_network_delay_ms(prev_node_for[stage.predecessor], node)
				
				# Add origin latency for first stage
				if not stage.predecessor and job.origin:
					origin_lat = self._compute_origin_latency(job, node)
					lat += origin_lat
				
				rel = self._reliability(node)
				score = rel - 0.001 * lat
				if score > best[1]:
					best = (node, score, exec_format)
			if best[0] is None:
				continue
			placements[stage.id] = PlacementDecision(stage_id=stage.id, node_name=best[0].name, exec_format=best[2])
			prev_node_for[stage.id] = best[0]
		return placements

"""Federated and fault-tolerant planner for the Fabric DT.

The :class:`FederatedPlanner` extends the baseline greedy policy with
network-aware scoring, federation-aware load balancing, and explicit
fallback placements so that the DT can survive correlated failures and
link partitions.

Key features
------------
* Penalises federations that are already saturated or degraded so new
  work is steered towards healthier domains.
* Accounts for link loss/latency when chaining stages, preferring nodes
  connected through resilient paths.
* Emits fallback assignments per-stage so operators can pre-warm or
  quickly fail over when chaos events hit a zone.
"""

# from __future__ import annotations  # Already at top

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from dt.cost_models import CostModels as CostModel
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    def merge_stage_details(*args): return {}
except ImportError:
    CostModel = object
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    def merge_stage_details(*args): return {}
from dt.state import DTState, safe_float

ModeConfig = Dict[str, Any]

DEFAULT_MODES: Dict[str, ModeConfig] = {
    "resilient": {
        "redundancy": 2,
        "risk_weight": 220.0,
        "load_weight": 380.0,
        "spread_weight": 210.0,
        "network_weight": 240.0,
        "resilience_weight": 250.0,
        "prefer_prev_bonus": 15.0,
        "reliability_weight": 360.0,
        "availability_penalty_ms": 260.0,
        "availability_horizon_sec": 240.0,
    },
    "network-aware": {
        "redundancy": 1,
        "risk_weight": 200.0,
        "load_weight": 260.0,
        "spread_weight": 140.0,
        "network_weight": 300.0,
        "resilience_weight": 190.0,
        "prefer_prev_bonus": 12.0,
        "reliability_weight": 320.0,
        "availability_penalty_ms": 220.0,
        "availability_horizon_sec": 180.0,
    },
    "federated": {
        "redundancy": 3,
        "risk_weight": 210.0,
        "load_weight": 360.0,
        "spread_weight": 260.0,
        "network_weight": 230.0,
        "resilience_weight": 240.0,
        "prefer_prev_bonus": 10.0,
        "reliability_weight": 340.0,
        "availability_penalty_ms": 240.0,
        "availability_horizon_sec": 300.0,
    },
}


def _mode_key(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode in DEFAULT_MODES:
        return mode
    if mode in ("fault-tolerant", "ft", "failover"):
        return "resilient"
    if mode in ("balanced", "load-balance", "load-balanced"):
        return "network-aware"
    return "resilient"


class FederatedPlanner:
    def __init__(self, state: DTState, cost_model: CostModel):
        self.state = state
        self.cm = cost_model

    def _adaptive_mode_cfg(self, base_cfg: ModeConfig, job: Dict[str, Any]) -> ModeConfig:
        cfg = dict(base_cfg)
        overview = self.state.predictive_overview()

        link_metrics = overview.get("links", {}) or {}
        latencies = [safe_float(v.get("latency_ms"), 0.0) for v in link_metrics.values() if v.get("latency_ms")]
        losses = [safe_float(v.get("loss_pct"), 0.0) for v in link_metrics.values() if v.get("loss_pct")]

        if latencies:
            avg_latency = sum(latencies) / max(1, len(latencies))
            if avg_latency > 20.0:
                cfg["network_weight"] = safe_float(cfg.get("network_weight"), 1.0) * 1.25
                cfg["spread_weight"] = safe_float(cfg.get("spread_weight"), 1.0) * 1.1
        if losses:
            avg_loss = sum(losses) / max(1, len(losses))
            if avg_loss > 1.5:
                cfg["resilience_weight"] = safe_float(cfg.get("resilience_weight"), 1.0) * 1.2

        node_metrics = overview.get("nodes", {}) or {}
        reliabilities = []
        availability_windows = []
        for entry in node_metrics.values():
            if entry.get("reliability") is not None:
                reliabilities.append(safe_float(entry.get("reliability"), 1.0))
            if entry.get("availability_window_sec") is not None:
                availability_windows.append(safe_float(entry.get("availability_window_sec"), 0.0))

        if reliabilities and (sum(reliabilities) / max(1, len(reliabilities))) < 0.85:
            cfg["reliability_weight"] = safe_float(cfg.get("reliability_weight"), 0.0) * 1.2 + 25.0
            cfg["resilience_weight"] = safe_float(cfg.get("resilience_weight"), 1.0) * 1.1

        if availability_windows and min(availability_windows) < safe_float(cfg.get("availability_horizon_sec"), 180.0):
            cfg["availability_penalty_ms"] = safe_float(cfg.get("availability_penalty_ms"), 200.0) * 1.15

        requested_redundancy = int(job.get("redundancy") or 1)
        cfg["redundancy"] = max(int(cfg.get("redundancy", 1)), requested_redundancy)

        return cfg

    # --------------------- helpers ---------------------

    def _supports_formats(self, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
        allowed = stage.get("allowed_formats") or []
        if not allowed:
            return True
        fmts = set(node.get("formats_supported") or [])
        return any(fmt in fmts for fmt in allowed)

    def _fits(self, node: Dict[str, Any], stage: Dict[str, Any]) -> bool:
        if (node.get("dyn") or {}).get("down", False):
            return False
        eff = node.get("effective") or {}
        res = stage.get("resources") or {}
        need_cpu = safe_float(res.get("cpu_cores"), 0.0)
        need_mem = safe_float(res.get("mem_gb"), 0.0)
        need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)
        if eff.get("free_cpu_cores", 0.0) + 1e-9 < need_cpu:
            return False
        if eff.get("free_mem_gb", 0.0) + 1e-9 < need_mem:
            return False
        if eff.get("free_gpu_vram_gb", 0.0) + 1e-9 < need_vram:
            return False
        return self._supports_formats(node, stage)

    def _choose_format(self, stage: Dict[str, Any], node: Dict[str, Any]) -> Optional[str]:
        allowed = stage.get("allowed_formats") or []
        if not allowed:
            return None
        fmts = node.get("formats_supported") or []
        for fmt in allowed:
            if fmt in fmts:
                return fmt
        return allowed[0] if allowed else None

    def _projected_load(
        self,
        entry: Dict[str, Any],
        need_cpu: float,
        need_mem: float,
        need_vram: float,
    ) -> float:
        loads: List[float] = []
        total_cpu = safe_float(entry.get("total_cpu_cores"), 0.0)
        if total_cpu > 0:
            free_cpu = max(0.0, safe_float(entry.get("free_cpu_cores"), 0.0) - need_cpu)
            loads.append(clamp((total_cpu - free_cpu) / max(total_cpu, 1e-6), 0.0, 1.0))

        total_mem = safe_float(entry.get("total_mem_gb"), 0.0)
        if total_mem > 0:
            free_mem = max(0.0, safe_float(entry.get("free_mem_gb"), 0.0) - need_mem)
            loads.append(clamp((total_mem - free_mem) / max(total_mem, 1e-6), 0.0, 1.0))

        total_vram = safe_float(entry.get("total_gpu_vram_gb"), 0.0)
        if total_vram > 0:
            free_vram = max(0.0, safe_float(entry.get("free_gpu_vram_gb"), 0.0) - need_vram)
            loads.append(clamp((total_vram - free_vram) / max(total_vram, 1e-6), 0.0, 1.0))

        if not loads:
            return 0.0
        return sum(loads) / len(loads)

    def _consume_resources(
        self,
        node: Dict[str, Any],
        fed_entry: Dict[str, Any],
        need_cpu: float,
        need_mem: float,
        need_vram: float,
    ) -> None:
        eff = node.setdefault("effective", {})
        eff["free_cpu_cores"] = max(0.0, safe_float(eff.get("free_cpu_cores"), 0.0) - need_cpu)
        eff["free_mem_gb"] = max(0.0, safe_float(eff.get("free_mem_gb"), 0.0) - need_mem)
        eff["free_gpu_vram_gb"] = max(0.0, safe_float(eff.get("free_gpu_vram_gb"), 0.0) - need_vram)

        fed_entry["free_cpu_cores"] = max(
            0.0, safe_float(fed_entry.get("free_cpu_cores"), 0.0) - need_cpu
        )
        fed_entry["free_mem_gb"] = max(
            0.0, safe_float(fed_entry.get("free_mem_gb"), 0.0) - need_mem
        )
        fed_entry["free_gpu_vram_gb"] = max(
            0.0, safe_float(fed_entry.get("free_gpu_vram_gb"), 0.0) - need_vram
        )
        # Recompute load factor for next stages
        fed_entry["load_factor"] = self._projected_load(fed_entry, 0.0, 0.0, 0.0)

    def _score_candidate(
        self,
        stage: Dict[str, Any],
        node_name: str,
        node: Dict[str, Any],
        federation: str,
        fed_entry: Dict[str, Any],
        prev_node: Optional[str],
        used_federations: Counter,
        mode_cfg: ModeConfig,
    ) -> Tuple[float, Dict[str, Any]]:
        res = stage.get("resources") or {}
        need_cpu = safe_float(res.get("cpu_cores"), 0.0)
        need_mem = safe_float(res.get("mem_gb"), 0.0)
        need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)

        projected_load = self._projected_load(fed_entry, need_cpu, need_mem, need_vram)

        fmt_override = self._choose_format(stage, node)
        stage_eval = dict(stage)
        if fmt_override:
            stage_eval["allowed_formats"] = [fmt_override]

        comp_ms = self.cm.compute_time_ms(stage_eval, node)
        energy_kj = self.cm.energy_kj(stage_eval, node, comp_ms)

        if prev_node in (None, node_name):
            xfer_ms = 0.0
            link_metrics = {"loss_pct": 0.0, "down": False, "rtt_ms": 0.0}
        else:
            xfer_ms = self.cm.transfer_time_ms(
                prev_node, node_name, safe_float(stage.get("size_mb"), 10.0)
            )
            link_metrics = self.state.effective_link_between(prev_node, node_name)

        link_loss = safe_float(link_metrics.get("loss_pct"), 0.0)
        risk = self.cm.risk_score(stage_eval, node, link_loss_pct=link_loss)

        load_penalty = mode_cfg["load_weight"] * projected_load
        spread_penalty = mode_cfg["spread_weight"] * used_federations[federation]
        network_penalty = mode_cfg["network_weight"] * (
            (1.0 if link_metrics.get("down") else 0.0)
            + clamp(link_loss / 10.0, 0.0, 1.0)
        )
        resilience_penalty = mode_cfg["resilience_weight"] * (
            safe_float(fed_entry.get("down_fraction"), 0.0)
            + safe_float(fed_entry.get("hot_fraction"), 0.0)
        )
        risk_penalty = mode_cfg["risk_weight"] * risk

        reliability = None
        availability = None
        try:
            reliability = self.state.node_reliability(node_name)
        except Exception:
            reliability = None
        try:
            availability = self.state.node_availability_window(node_name)
        except Exception:
            availability = None

        reliability_penalty = 0.0
        reliability_weight = safe_float(mode_cfg.get("reliability_weight"), 0.0)
        if reliability is not None and reliability_weight > 0.0:
            reliability_penalty = reliability_weight * max(0.0, 1.0 - clamp(float(reliability), 0.0, 1.0))

        availability_penalty = 0.0
        availability_weight = safe_float(mode_cfg.get("availability_penalty_ms"), 0.0)
        horizon = safe_float(mode_cfg.get("availability_horizon_sec"), 0.0)
        if availability is not None and availability_weight > 0.0 and horizon > 0.0:
            avail = max(0.0, float(availability))
            availability_penalty = availability_weight * max(0.0, (horizon - min(avail, horizon)) / horizon)

        score = (
            comp_ms
            + xfer_ms
            + load_penalty
            + spread_penalty
            + network_penalty
            + resilience_penalty
            + risk_penalty
            + reliability_penalty
            + availability_penalty
        )

        if prev_node and prev_node == node_name:
            score -= mode_cfg["prefer_prev_bonus"]

        metrics = {
            "format": fmt_override,
            "compute_ms": round(comp_ms, 3),
            "xfer_ms": round(xfer_ms, 3),
            "energy_kj": round(energy_kj, 5),
            "risk": round(risk, 4),
            "reliability": None if reliability is None else round(float(reliability), 4),
            "availability_window_sec": None
            if availability is None
            else round(float(availability), 3),
            "score": round(score, 3),
            "load_penalty_ms": round(load_penalty, 3),
            "network_penalty_ms": round(network_penalty, 3),
            "resilience_penalty_ms": round(resilience_penalty, 3),
            "projected_load": round(projected_load, 4),
            "link_loss_pct": round(link_loss, 4),
            "reliability_penalty_ms": round(reliability_penalty, 3),
            "availability_penalty_ms": round(availability_penalty, 3),
        }

        return score, metrics

    # --------------------- public ---------------------

    def plan_job(
        self,
        job: Dict[str, Any],
        dry_run: bool = False,
        mode: str = "resilient",
    ) -> Dict[str, Any]:
        stages = job.get("stages") or []
        if not stages:
            return {
                "job_id": job.get("id"),
                "assignments": {},
                "per_stage": [],
                "reservations": [],
                "shadow_assignments": {},
                "latency_ms": 0.0,
                "energy_kj": 0.0,
                "risk": 0.0,
                "strategy": mode,
                "dry_run": dry_run,
                "infeasible": True,
                "reason": "no_stages",
                "ts": int(time.time() * 1000),
            }

        cfg = self._adaptive_mode_cfg(DEFAULT_MODES[_mode_key(mode)], job)

        nodes = self.state.nodes_for_planner()
        fed_overview = self.state.federations_overview()
        fed_stats_map = {
            entry["name"]: dict(entry)
            for entry in (fed_overview.get("federations") or [])
        }
        node_to_fed = fed_overview.get("node_federations") or {}

        assignments: Dict[str, str] = {}
        shadow_assignments: Dict[str, List[str]] = {}
        per_stage: List[Dict[str, Any]] = []
        reservations: List[Dict[str, str]] = []

        used_federations: Counter = Counter()
        prev_node: Optional[str] = None
        infeasible = False
        fallback_crossfed = 0
        stage_resilience_scores: List[float] = []

        for stage in stages:
            sid = stage.get("id")
            if not sid:
                continue

            res = stage.get("resources") or {}
            need_cpu = safe_float(res.get("cpu_cores"), 0.0)
            need_mem = safe_float(res.get("mem_gb"), 0.0)
            need_vram = safe_float(res.get("gpu_vram_gb"), 0.0)

            candidates: List[Tuple[float, Dict[str, Any], str, Dict[str, Any]]] = []

            for node_name, node in nodes.items():
                if not self._fits(node, stage):
                    continue
                federation = node_to_fed.get(node_name) or self.state.federation_for_node(node_name) or "global"
                fed_entry = fed_stats_map.setdefault(
                    federation,
                    {
                        "name": federation,
                        "total_cpu_cores": safe_float((node.get("caps") or {}).get("max_cpu_cores"), 0.0),
                        "free_cpu_cores": safe_float((node.get("effective") or {}).get("free_cpu_cores"), 0.0),
                        "total_mem_gb": safe_float((node.get("caps") or {}).get("ram_gb"), 0.0),
                        "free_mem_gb": safe_float((node.get("effective") or {}).get("free_mem_gb"), 0.0),
                        "total_gpu_vram_gb": safe_float((node.get("caps") or {}).get("gpu_vram_gb"), 0.0),
                        "free_gpu_vram_gb": safe_float((node.get("effective") or {}).get("free_gpu_vram_gb"), 0.0),
                        "down_fraction": 0.0,
                        "hot_fraction": 0.0,
                        "load_factor": 0.0,
                    },
                )

                score, metrics = self._score_candidate(
                    stage,
                    node_name,
                    node,
                    federation,
                    fed_entry,
                    prev_node,
                    used_federations,
                    cfg,
                )
                candidates.append((score, metrics, node_name, fed_entry))

            if not candidates:
                infeasible = True
                per_stage.append(
                    {
                        "id": sid,
                        "node": None,
                        "infeasible": True,
                        "reason": "no_feasible_node",
                    }
                )
                prev_node = None
                continue

            candidates.sort(key=lambda item: item[0])
            best_score, best_metrics, best_name, best_fed_entry = candidates[0]
            best_node = nodes.get(best_name, {})
            best_fed_name = node_to_fed.get(best_name) or best_fed_entry.get("name") or "global"

            fallback_nodes: List[str] = []
            fallback_feds: List[str] = []
            fallback_entries: List[Dict[str, Any]] = []
            redundancy = max(1, int(cfg["redundancy"]))
            target_fallbacks = max(0, redundancy - 1)

            if target_fallbacks > 0 and len(candidates) > 1:
                extra: List[Tuple[float, Dict[str, Any], str, Dict[str, Any]]] = []
                preferred: List[Tuple[float, Dict[str, Any], str, Dict[str, Any]]] = []
                for candidate in candidates[1:]:
                    cand_name = candidate[2]
                    cand_fed = node_to_fed.get(cand_name) or candidate[3].get("name") or "global"
                    if cand_fed != best_fed_name:
                        preferred.append(candidate)
                    else:
                        extra.append(candidate)
                ordered = preferred + extra
                for candidate in ordered:
                    cand_name = candidate[2]
                    cand_fed = node_to_fed.get(cand_name) or candidate[3].get("name") or "global"
                    cand_metrics = candidate[1]
                    entry = {
                        "node": cand_name,
                        "score": round(candidate[0], 3),
                        "reliability": cand_metrics.get("reliability"),
                        "availability_window_sec": cand_metrics.get("availability_window_sec"),
                        "federation": cand_fed,
                    }
                    fallback_entries.append(entry)
                    fallback_nodes.append(cand_name)
                    fallback_feds.append(cand_fed)
                    if cand_fed != best_fed_name:
                        fallback_crossfed += 1
                    if len(fallback_entries) >= target_fallbacks:
                        break

            shadow_assignments[sid] = list(fallback_nodes)

            res_id = None
            assigned = True
            if not dry_run:
                req = {
                    "node": best_name,
                    "cpu_cores": need_cpu,
                    "mem_gb": need_mem,
                    "gpu_vram_gb": need_vram,
                }
                res_id = self.state.reserve(req)
                if res_id is None:
                    assigned = False
                    infeasible = True

            shadow_records: List[Dict[str, Any]] = []
            if assigned:
                assignments[sid] = best_name
                if res_id:
                    reservations.append(
                        {
                            "node": best_name,
                            "reservation_id": res_id,
                            "stage_id": sid,
                            "role": "primary",
                        }
                    )
                self._consume_resources(best_node, best_fed_entry, need_cpu, need_mem, need_vram)
                used_federations[best_fed_name] += 1
                prev_node = best_name
            else:
                prev_node = None

            stage_record = {
                "id": sid,
                "node": best_name if assigned else None,
                "reservation_id": res_id,
                "federation": best_fed_name,
                "fallbacks": fallback_entries,
                "fallback_federations": fallback_feds,
                "infeasible": not assigned,
                **best_metrics,
            }

            if assigned and not dry_run and target_fallbacks > 0:
                for fb_entry in fallback_entries:
                    fb_node = fb_entry.get("node")
                    if not fb_node or fb_node == best_name:
                        continue
                    req = {
                        "node": fb_node,
                        "cpu_cores": need_cpu,
                        "mem_gb": need_mem,
                        "gpu_vram_gb": need_vram,
                    }
                    fb_res_id = self.state.reserve(req)
                    if not fb_res_id:
                        continue
                    shadow_records.append({"node": fb_node, "reservation_id": fb_res_id})
                    reservations.append(
                        {
                            "node": fb_node,
                            "reservation_id": fb_res_id,
                            "stage_id": sid,
                            "role": "shadow",
                        }
                    )
                    if len(shadow_records) >= target_fallbacks:
                        break

            if shadow_records:
                stage_record["shadow_reservations"] = shadow_records
            if fallback_entries:
                stage_record.setdefault("fallback_summaries", fallback_entries)
            per_stage.append(stage_record)

            # Estimate stage resilience using node reliability and fallback diversity
            primary_rel = clamp(safe_float(best_metrics.get("reliability"), 0.72), 0.05, 0.995)
            horizon = safe_float(cfg.get("availability_horizon_sec"), 240.0)
            primary_avail = safe_float(best_metrics.get("availability_window_sec"), horizon)
            if horizon > 0.0 and primary_avail > 0.0 and primary_avail < horizon:
                primary_rel *= max(0.35, primary_avail / horizon)
            risk_penalty = safe_float(best_metrics.get("risk"), 0.0)
            if risk_penalty and risk_penalty > 0.5:
                primary_rel *= max(0.25, 1.0 - min(0.9, risk_penalty))

            fallback_failure = 1.0
            for fb_entry in fallback_entries:
                fb_rel = clamp(safe_float(fb_entry.get("reliability"), 0.5), 0.05, 0.95)
                fb_avail = safe_float(fb_entry.get("availability_window_sec"), horizon)
                if horizon > 0.0 and fb_avail > 0.0 and fb_avail < horizon * 0.5:
                    fb_rel *= 0.7
                if fb_entry.get("federation") == best_fed_name:
                    fb_rel *= 0.65
                fallback_failure *= max(0.05, 1.0 - fb_rel)

            stage_resilience = 1.0 - ((1.0 - clamp(primary_rel, 0.05, 0.995)) * fallback_failure)
            stage_resilience = clamp(stage_resilience, 0.0, 0.995)
            stage_record["resilience_estimate"] = round(stage_resilience, 4)
            stage_resilience_scores.append(stage_resilience)

        cost = self.cm.job_cost(job, assignments)
        merged = merge_stage_details(per_stage, cost.get("per_stage"))
        ddl = safe_float(job.get("deadline_ms"), 0.0)
        slo_penalty = self.cm.slo_penalty(ddl, cost.get("latency_ms", float("inf"))) if ddl > 0 else 0.0

        unique_feds = {
            node_to_fed.get(node) or self.state.federation_for_node(node) or "global"
            for node in assignments.values()
        }
        spread = len(unique_feds) / max(1, len(stages))
        fallback_ratio = sum(1 for v in shadow_assignments.values() if v) / max(1, len(stages))
        crossfed_ratio = fallback_crossfed / max(1, len(stages))

        projected_feds = []
        for entry in fed_stats_map.values():
            projected_feds.append(
                {
                    "name": entry.get("name"),
                    "free_cpu_cores": round(safe_float(entry.get("free_cpu_cores"), 0.0), 4),
                    "free_mem_gb": round(safe_float(entry.get("free_mem_gb"), 0.0), 4),
                    "free_gpu_vram_gb": round(safe_float(entry.get("free_gpu_vram_gb"), 0.0), 4),
                    "load_factor": round(safe_float(entry.get("load_factor"), 0.0), 4),
                }
            )
        projected_feds.sort(key=lambda x: x["name"] or "")

        avg_resilience = (
            sum(stage_resilience_scores) / max(1, len(stage_resilience_scores))
            if stage_resilience_scores else 0.0
        )

        result = {
            "job_id": job.get("id"),
            "assignments": assignments,
            "reservations": reservations,
            "shadow_assignments": shadow_assignments,
            "per_stage": merged,
            "latency_ms": cost.get("latency_ms"),
            "energy_kj": cost.get("energy_kj"),
            "risk": cost.get("risk"),
            "avg_reliability": cost.get("avg_reliability"),
            "deadline_ms": ddl or None,
            "slo_penalty": slo_penalty,
            "infeasible": infeasible or (cost.get("latency_ms") == float("inf")),
            "strategy": mode,
            "dry_run": dry_run,
            "federation_spread": round(spread, 4),
            "federations_in_use": sorted(unique_feds),
            "fallback_coverage": round(fallback_ratio, 4),
            "resilience_score": round(clamp(avg_resilience, 0.0, 0.999), 4),
            "cross_federation_fallback_ratio": round(crossfed_ratio, 4),
            "projected_federations": projected_feds,
            "ts": int(time.time() * 1000),
        }

        return result

