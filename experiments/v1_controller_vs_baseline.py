from __future__ import annotations

import time
import json
import requests
from typing import Any, Dict


def make_job(job_id: int, delay_ms: int) -> Dict[str, Any]:
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {"name": f"job-{job_id}", "deadline_ms": delay_ms * 2},
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


def run(dt_url: str, n: int = 10) -> None:
	for i in range(n):
		job = make_job(i, 1000 + 50 * i)
		resp = requests.post(f"{dt_url}/plan", json={"job": job, "strategy": "resilient", "dry_run": False}, timeout=20)
		resp.raise_for_status()
		print(json.dumps(resp.json()))


if __name__ == "__main__":
	run("http://localhost:8080", 5)





