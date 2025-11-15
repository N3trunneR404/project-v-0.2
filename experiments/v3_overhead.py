"""V3: Overhead Experiment

Measures execution format overhead (native vs emulated).
Includes origin context and resource scaling documentation.
"""

from __future__ import annotations

import json
import requests
from typing import Any, Dict, Optional

# Resource scaling: default 1:100 (1 simulated CPU = 0.01 real cores)
RESOURCE_SCALE = 0.01


def mk(
	stage_format: str,
	ms: int,
	origin_cluster: str = "dc-core",
	origin_node: Optional[str] = None
) -> Dict[str, Any]:
	"""Create a job with origin context."""
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {
			"name": f"overhead-{stage_format}",
			"deadline_ms": ms * 3,
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
					"constraints": {"arch": ["amd64"], "formats": ["native", "wasm"]},
				}
			]
		},
	}


def run(dt_url: str) -> None:
	"""Run experiment with origin context and scaling."""
	print(f"=== V3 Overhead ===")
	print(f"Resource scale: {RESOURCE_SCALE} (1:{int(1.0/RESOURCE_SCALE)})")
	print()
	
	for fmt in ["native", "wasm"]:
		job = mk(fmt, 1500, origin_cluster="dc-core")
		resp = requests.post(
			f"{dt_url}/plan",
			json={"job": job, "strategy": "cvar", "dry_run": True},
			timeout=20
		)
		resp.raise_for_status()
		result = resp.json()
		result["origin_cluster"] = "dc-core"
		result["resource_scale"] = RESOURCE_SCALE
		print(json.dumps(result))


if __name__ == "__main__":
	run("http://localhost:8080")





