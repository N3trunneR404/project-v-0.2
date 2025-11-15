"""V2: Predictive Ablation Experiment

Tests predictive capabilities with different strategies.
Includes origin context and resource scaling documentation.
"""

from __future__ import annotations

import json
import random
import requests
from typing import Any, Dict, Optional

# Resource scaling: default 1:100 (1 simulated CPU = 0.01 real cores)
RESOURCE_SCALE = 0.01


def make_job(
	job_id: int,
	origin_cluster: str = "campus-lab",
	origin_node: Optional[str] = None
) -> Dict[str, Any]:
	"""Create a job with origin context."""
	dur = random.randint(800, 1600)
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {
			"name": f"ablate-{job_id}",
			"deadline_ms": dur * 2,
			"origin": {
				"cluster": origin_cluster,
				"node": origin_node,
			}
		},
		"spec": {
			"stages": [
				{
					"id": "s1",
					"compute": {"cpu": 1, "mem_gb": 1, "duration_ms": dur},
					"constraints": {"arch": ["amd64"], "formats": ["native"]},
				}
			]
		},
	}


def run(dt_url: str, trials: int = 20) -> None:
	"""Run experiment with origin context and scaling."""
	print(f"=== V2 Predictive Ablation ===")
	print(f"Resource scale: {RESOURCE_SCALE} (1:{int(1.0/RESOURCE_SCALE)})")
	print(f"Trials: {trials}")
	print()
	
	strategies = ["greedy", "resilient", "cvar"]
	origin_clusters = ["dc-core", "edge-microdc", "campus-lab", "gamer-pc"]
	
	for i in range(trials):
		# Rotate origin clusters
		origin = origin_clusters[i % len(origin_clusters)]
		job = make_job(i, origin_cluster=origin)
		strategy = strategies[i % len(strategies)]
		
		resp = requests.post(
			f"{dt_url}/plan",
			json={"job": job, "strategy": strategy, "dry_run": True},
			timeout=20
		)
		resp.raise_for_status()
		result = resp.json()
		result["origin_cluster"] = origin
		result["resource_scale"] = RESOURCE_SCALE
		print(json.dumps(result))


if __name__ == "__main__":
	run("http://localhost:8080", 10)





