"""V5: Scalability KWOK Experiment

Tests DT scalability with large number of jobs.
Includes origin context and resource scaling documentation.
"""

from __future__ import annotations

import time
import requests
from typing import Optional

# Resource scaling: default 1:100 (1 simulated CPU = 0.01 real cores)
RESOURCE_SCALE = 0.01


def run(dt_url: str, jobs: int = 200) -> None:
	"""Run scalability experiment with origin context and scaling."""
	print(f"=== V5 Scalability KWOK ===")
	print(f"Resource scale: {RESOURCE_SCALE} (1:{int(1.0/RESOURCE_SCALE)})")
	print(f"Jobs: {jobs}")
	print()
	
	start = time.time()
	origin_clusters = ["dc-core", "edge-microdc", "campus-lab", "phone-pan-1", "phone-pan-2"]
	
	for i in range(jobs):
		# Rotate origin clusters
		origin = origin_clusters[i % len(origin_clusters)]
		j = {
			"apiVersion": "fabric.dt/v1",
			"kind": "Job",
			"metadata": {
				"name": f"scale-{i}",
				"deadline_ms": 1500,
				"origin": {
					"cluster": origin,
					"node": None,
				}
			},
			"spec": {
				"stages": [{
					"id": "s1",
					"compute": {"cpu": 1, "mem_gb": 1, "duration_ms": 800},
					"constraints": {"arch": ["amd64"], "formats": ["native"]}
				}]
			},
		}
		requests.post(
			f"{dt_url}/plan",
			json={"job": j, "strategy": "greedy", "dry_run": True},
			timeout=15
		)
	
	elapsed = (time.time() - start) * 1000.0
	print(f"jobs={jobs} total_ms={elapsed:.1f} avg_ms_per_job={elapsed/jobs:.1f} resource_scale={RESOURCE_SCALE}")


if __name__ == "__main__":
	run("http://localhost:8080", 200)





