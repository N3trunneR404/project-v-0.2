"""V6: Drift Robustness Experiment

Tests DT robustness to workload drift over time.
Includes origin context and resource scaling documentation.
"""

from __future__ import annotations

import json
import random
import requests
from typing import Any, Dict, Optional

# Resource scaling: default 1:100 (1 simulated CPU = 0.01 real cores)
RESOURCE_SCALE = 0.01


def mk_job(
	name: str,
	ms: int,
	origin_cluster: str = "campus-lab",
	origin_node: Optional[str] = None
) -> Dict[str, Any]:
	"""Create a job with origin context."""
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {
			"name": name,
			"deadline_ms": ms * 2,
			"origin": {
				"cluster": origin_cluster,
				"node": origin_node,
			}
		},
		"spec": {
			"stages": [
				{
					"id": "s1",
					"compute": {"cpu": 1, "mem_gb": 1, "duration_ms": ms},
					"constraints": {"arch": ["amd64"], "formats": ["native"]},
				}
			]
		},
	}


def run(dt_url: str, phases: int = 3, per_phase: int = 5) -> None:
	"""Run experiment with origin context and scaling."""
	print(f"=== V6 Drift Robustness ===")
	print(f"Resource scale: {RESOURCE_SCALE} (1:{int(1.0/RESOURCE_SCALE)})")
	print(f"Phases: {phases}, Jobs per phase: {per_phase}")
	print()
	
	origin_clusters = ["dc-core", "edge-microdc", "campus-lab", "gamer-pc"]
	
	for p in range(phases):
		scale = 1.0 + 0.25 * p
		for i in range(per_phase):
			ms = int(random.randint(800, 1400) * scale)
			# Rotate origin clusters
			origin = origin_clusters[(p * per_phase + i) % len(origin_clusters)]
			job = mk_job(f"drift-{p}-{i}", ms, origin_cluster=origin)
			resp = requests.post(
				f"{dt_url}/plan",
				json={"job": job, "strategy": "greedy", "dry_run": True},
				timeout=20
			)
			resp.raise_for_status()
			result = resp.json()
			result["phase"] = p
			result["origin_cluster"] = origin
			result["resource_scale"] = RESOURCE_SCALE
			print(json.dumps(result))


if __name__ == "__main__":
	run("http://localhost:8080", 3, 5)





