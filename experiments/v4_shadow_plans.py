"""V4: Shadow Plans Experiment

Tests shadow plan generation for failover scenarios.
Includes origin context, resource scaling, and verification collection.
"""

from __future__ import annotations

import json
import requests
from typing import Any, Dict, Optional

# Resource scaling: default 1:100 (1 simulated CPU = 0.01 real cores)
RESOURCE_SCALE = 0.01


def job(
	deadline_ms: int,
	origin_cluster: str = "edge-microdc",
	origin_node: Optional[str] = None
) -> Dict[str, Any]:
	"""Create a job with origin context."""
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {
			"name": "shadow-test",
			"deadline_ms": deadline_ms,
			"origin": {
				"cluster": origin_cluster,
				"node": origin_node,
			}
		},
		"spec": {
			"stages": [
				{
					"id": "s1",
					"compute": {"cpu": 1, "mem_gb": 1, "duration_ms": deadline_ms // 2},
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


def run(dt_url: str) -> None:
	"""Run experiment with origin context, scaling, and verification."""
	print(f"=== V4 Shadow Plans ===")
	print(f"Resource scale: {RESOURCE_SCALE} (1:{int(1.0/RESOURCE_SCALE)})")
	print()
	
	job_spec = job(2000, origin_cluster="edge-microdc")
	resp = requests.post(
		f"{dt_url}/plan",
		json={"job": job_spec, "strategy": "resilient", "dry_run": False},
		timeout=20
	)
	resp.raise_for_status()
	result = resp.json()
	
	plan_id = result.get("plan_id")
	result["origin_cluster"] = "edge-microdc"
	result["resource_scale"] = RESOURCE_SCALE
	
	# Collect verification if available
	if plan_id:
		verify_data = collect_verification(dt_url, plan_id)
		if verify_data:
			result["verification"] = verify_data
	
	print(json.dumps(result))


if __name__ == "__main__":
	run("http://localhost:8080")





