from __future__ import annotations

import uuid
from typing import Any, Dict

from flask import Flask, jsonify, request

from typing import Optional

from dt.actuator import Actuator
from dt.predict import PredictiveSimulator
from dt.state import DTState, Job, JobStage, StageCompute, StageConstraints
from dt.policy.greedy import GreedyLatencyPolicy
from dt.policy.resilient import ResilientPolicy
from dt.policy.cvar import RiskAwareCvarPolicy
from dt.cluster_manager import ClusterManager


def create_app(state: DTState, cluster_manager: Optional[ClusterManager] = None) -> Flask:
	app = Flask(__name__)
	# Store state in app config so it's accessible in all endpoints
	app.config['dt_state'] = state
	app.config['cluster_manager'] = cluster_manager
	
	sim = PredictiveSimulator(state)
	actuator = Actuator(cluster_manager=cluster_manager)

	def select_policy(name: str):
		name = (name or "greedy").lower()
		if name == "resilient":
			return ResilientPolicy(state, sim, cluster_manager=cluster_manager)
		if name == "cvar":
			return RiskAwareCvarPolicy(state, sim, cluster_manager=cluster_manager)
		return GreedyLatencyPolicy(state, sim, cluster_manager=cluster_manager)

	@app.post("/plan")
	def plan() -> Any:
		state = app.config['dt_state']
		# Safety check: ensure state is seeded
		nodes = state.list_nodes()
		if not nodes:
			try:
				from app import seed_state
				seed_state(state)
			except Exception:
				pass
		
		body: Dict[str, Any] = request.get_json(force=True)
		job_spec = body.get("job")
		if not job_spec:
			return jsonify({"error": "missing job spec"}), 400
		strategy = body.get("strategy", "greedy")
		dry_run = bool(body.get("dry_run", False))

		job = _job_from_dict(job_spec)
		policy = select_policy(strategy)
		placements = policy.place(job)
		if not placements:
			return jsonify({"error": "no feasible placements found", "stages": [s.id for s in job.stages]}), 400
		metrics = sim.score_plan(job, placements)
		plan_id = f"plan-{uuid.uuid4().hex[:8]}"
		response = {
			"plan_id": plan_id,
			"placements": {
				stage_id: {
					"stage_id": stage_id,
					"node_name": decision.node_name,
					"exec_format": decision.exec_format,
				}
				for stage_id, decision in placements.items()
			},
			"predicted_latency_ms": metrics.latency_ms,
			"predicted_energy_kwh": metrics.energy_kwh,
			"risk_score": metrics.risk_score,
			"shadow_plan": {f"{sid}_backup": dec.node_name for sid, dec in placements.items()},
		}
		if not dry_run:
			try:
				actuator.submit_plan(job, placements, plan_id=plan_id)
			except Exception as e:
				# Log error but don't fail the API response
				# The plan was already computed and returned
				import logging
				logging.getLogger(__name__).error(f"Failed to submit plan {plan_id}: {e}")
		return jsonify(response)

	@app.post("/observe")
	def observe() -> Any:
		state = app.config['dt_state']
		body: Dict[str, Any] = request.get_json(force=True) or {}
		etype = body.get("type")
		node = body.get("node")
		
		if not etype or not node:
			return jsonify({"error": "missing 'type' or 'node' field"}), 400
		
		try:
			if etype == "node_down":
				state.mark_node_availability(node, False)
			elif etype == "node_up":
				state.mark_node_availability(node, True)
			else:
				return jsonify({"error": f"unknown event type: {etype}"}), 400
			return jsonify({"status": "ok", "node": node, "event": etype})
		except Exception as e:
			return jsonify({"error": str(e)}), 500

	@app.get("/snapshot")
	def snapshot() -> Any:
		state = app.config['dt_state']
		# Safety check: ensure state is seeded
		nodes = state.list_nodes()
		if not nodes:
			# Try to seed if not already seeded (for worker isolation)
			try:
				from app import seed_state
				seed_state(state)
				nodes = state.list_nodes()
			except Exception:
				pass
		return jsonify({"nodes": [n.name for n in nodes]})

	@app.get("/plan/<plan_id>/verify")
	def verify_plan(plan_id: str) -> Any:
		"""Get verification results for a plan."""
		state = app.config['dt_state']
		
		# Get observed metrics
		observed = state.get_observed_metrics(plan_id)
		if not observed:
			return jsonify({"error": f"No observed metrics found for plan {plan_id}"}), 404
		
		# Return observed metrics
		# Note: In full implementation, predicted metrics would be retrieved from stored plan
		return jsonify({
			"plan_id": plan_id,
			"observed": {
				"latency_ms": observed.latency_ms,
				"cpu_util": observed.cpu_util,
				"mem_peak_gb": observed.mem_peak_gb,
				"energy_kwh": observed.energy_kwh,
				"completed_at": observed.completed_at,
			},
			"note": "Predicted values should be retrieved from stored plan in full implementation"
		})

	return app


def _job_from_dict(job: Dict[str, Any]) -> Job:
	"""Parse job from dictionary, including origin context."""
	stages = []
	for stage_spec in job["spec"]["stages"]:
		compute = StageCompute(
			cpu=int(stage_spec["compute"]["cpu"]),
			mem_gb=int(stage_spec["compute"]["mem_gb"]),
			duration_ms=int(stage_spec["compute"]["duration_ms"]),
			gpu_vram_gb=int(stage_spec["compute"].get("gpu_vram_gb", 0)),
			workload_type=stage_spec["compute"].get("workload_type", "cpu_bound"),
		)
		constraints = StageConstraints(
			arch=list(stage_spec["constraints"].get("arch", ["amd64"])),
			formats=list(stage_spec["constraints"].get("formats", ["native"])),
			data_locality=stage_spec["constraints"].get("data_locality"),
			max_latency_to_predecessor_ms=stage_spec["constraints"].get("max_latency_to_predecessor_ms"),
		)
		stages.append(
			JobStage(
				id=stage_spec["id"],
				compute=compute,
				constraints=constraints,
				predecessor=stage_spec.get("predecessor"),
			)
		)
	# Parse origin context
	origin = None
	if "origin" in job.get("metadata", {}):
		origin_dict = job["metadata"]["origin"]
		from dt.state import JobOrigin
		origin = JobOrigin(
			cluster=origin_dict.get("cluster", "dc-core"),
			node=origin_dict.get("node"),
		)
	
	return Job(
		name=job["metadata"]["name"],
		deadline_ms=int(job["metadata"].get("deadline_ms", 60000)),  # Default 60s if not specified
		stages=stages,
		origin=origin,
	)
