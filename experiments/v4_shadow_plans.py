from __future__ import annotations

import json
import requests
from typing import Any, Dict


def job(deadline_ms: int) -> Dict[str, Any]:
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {"name": "shadow-test", "deadline_ms": deadline_ms},
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


def run(dt_url: str) -> None:
	resp = requests.post(f"{dt_url}/plan", json={"job": job(2000), "strategy": "resilient", "dry_run": False}, timeout=20)
	resp.raise_for_status()
	print(json.dumps(resp.json()))


if __name__ == "__main__":
	run("http://localhost:8080")





