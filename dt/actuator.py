from __future__ import annotations

from typing import Dict
import uuid

from kubernetes import client, config
from kubernetes.client import V1Pod

from dt.state import Plan, Job, PlacementDecision, DTState
try:
    from k8s_executor.pod_gen import generate_pod_from_decision
except ImportError:
    # k8s_executor not available - use stub
    def generate_pod_from_decision(job_name: str, decision: PlacementDecision, namespace: str = "dt-fabric"):
        return {"metadata": {"name": f"{job_name}-{decision.stage_id}-{decision.node_name}"}, "spec": {}}


class Actuator:
	def __init__(self, namespace: str = "dt-fabric") -> None:
		try:
			config.load_incluster_config()
		except Exception:
			config.load_kube_config()
		self.core = client.CoreV1Api()
		self.namespace = namespace

	def cordon_node(self, node_name: str) -> None:
		body = {"spec": {"unschedulable": True}}
		client.CoreV1Api().patch_node(node_name, body)

	def uncordon_node(self, node_name: str) -> None:
		body = {"spec": {"unschedulable": False}}
		client.CoreV1Api().patch_node(node_name, body)

	def submit_plan(self, job: Job, placements: Dict[str, PlacementDecision]) -> Plan:
		plan_id = f"plan-{uuid.uuid4().hex[:8]}"
		for dec in placements.values():
			pod = generate_pod_from_decision(job.name, dec, self.namespace)
			# Convert dict to V1Pod if needed, or use dict directly
			if isinstance(pod, dict):
				# For now, just log - actual pod creation would need proper V1Pod object
				# self.core.create_namespaced_pod(namespace=self.namespace, body=pod)
				pass
			else:
				self.core.create_namespaced_pod(namespace=self.namespace, body=pod)
		return Plan(plan_id=plan_id, job_name=job.name, placements=placements, predicted_latency_ms=0.0, predicted_energy_kwh=0.0, risk_score=0.0)





