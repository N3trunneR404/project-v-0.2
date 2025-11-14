from __future__ import annotations

import json
import random
import requests
from typing import Any, Dict


def mk_job(name: str, ms: int) -> Dict[str, Any]:
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {"name": name, "deadline_ms": ms * 2},
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
	for p in range(phases):
		scale = 1.0 + 0.25 * p
		for i in range(per_phase):
			ms = int(random.randint(800, 1400) * scale)
			job = mk_job(f"drift-{p}-{i}", ms)
			resp = requests.post(f"{dt_url}/plan", json={"job": job, "strategy": "greedy", "dry_run": True}, timeout=20)
			resp.raise_for_status()
			print(json.dumps(resp.json()))


if __name__ == "__main__":
	run("http://localhost:8080", 3, 5)





