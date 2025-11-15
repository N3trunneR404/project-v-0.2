"""V1: Controller vs Baseline Experiment

Compares DT controller with baseline scheduling strategies.
Includes origin context, resource scaling, and verification collection.
"""

from __future__ import annotations

import time
import json
import requests
from typing import Any, Dict, List, Optional

# Resource scaling: default 1:100 (1 simulated CPU = 0.01 real cores)
RESOURCE_SCALE = 0.01


def make_job(
	job_id: int,
	delay_ms: int,
	origin_cluster: str = "dc-core",
	origin_node: Optional[str] = None
) -> Dict[str, Any]:
	"""Create a job with origin context and resource scaling."""
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {
			"name": f"job-{job_id}",
			"deadline_ms": delay_ms * 2,
			"origin": {
				"cluster": origin_cluster,
				"node": origin_node,
			}
		},
		"spec": {
			"stages": [
				{
					"id": "s1",
					"compute": {"cpu": 1, "mem_gb": 1, "duration_ms": delay_ms},
					"constraints": {"arch": ["amd64"], "formats": ["native"]},
				}
			]
		},
	}


def collect_verification(dt_url: str, plan_id: str) -> Optional[Dict[str, Any]]:
	"""Collect verification results for a plan."""
	try:
		resp = requests.get(f"{dt_url}/plan/{plan_id}/verify", timeout=10)
		if resp.status_code == 200:
			return resp.json()
	except Exception:
		pass
	return None


def run(dt_url: str, n: int = 10) -> None:
	"""Run experiment with origin context, scaling, and verification."""
	print(f"=== V1 Controller vs Baseline ===")
	print(f"Resource scale: {RESOURCE_SCALE} (1:{int(1.0/RESOURCE_SCALE)})")
	print(f"Jobs: {n}")
	print()
	
	results: List[Dict[str, Any]] = []
	origin_clusters = ["dc-core", "edge-microdc", "campus-lab"]
	
	for i in range(n):
		# Rotate origin clusters
		origin = origin_clusters[i % len(origin_clusters)]
		job = make_job(i, 1000 + 50 * i, origin_cluster=origin)
		
		resp = requests.post(
			f"{dt_url}/plan",
			json={"job": job, "strategy": "resilient", "dry_run": False},
			timeout=20
		)
		resp.raise_for_status()
		plan_data = resp.json()
		
		plan_id = plan_data.get("plan_id")
		result = {
			"job_id": i,
			"plan_id": plan_id,
			"origin_cluster": origin,
			"predicted_latency_ms": plan_data.get("predicted_latency_ms"),
			"predicted_energy_kwh": plan_data.get("predicted_energy_kwh"),
			"resource_scale": RESOURCE_SCALE,
		}
		
		# Collect verification if available
		if plan_id:
			verify_data = collect_verification(dt_url, plan_id)
			if verify_data:
				result["verification"] = verify_data
		
		results.append(result)
		print(json.dumps(result))
	
	# Summary
	print()
	print(f"=== Summary ===")
	print(f"Total jobs: {len(results)}")
	print(f"Resource scale: {RESOURCE_SCALE} (1:{int(1.0/RESOURCE_SCALE)})")


if __name__ == "__main__":
	run("http://localhost:8080", 5)





