from __future__ import annotations

import json
import random
import requests
from typing import Any, Dict


def make_job(job_id: int) -> Dict[str, Any]:
	dur = random.randint(800, 1600)
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {"name": f"ablate-{job_id}", "deadline_ms": dur * 2},
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
	for i in range(trials):
		job = make_job(i)
		resp = requests.post(f"{dt_url}/plan", json={"job": job, "strategy": "greedy", "dry_run": True}, timeout=20)
		resp.raise_for_status()
		print(json.dumps(resp.json()))


if __name__ == "__main__":
	run("http://localhost:8080", 10)





