from __future__ import annotations

import json
import requests
from typing import Any, Dict


def mk(stage_format: str, ms: int) -> Dict[str, Any]:
	return {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {"name": f"overhead-{stage_format}", "deadline_ms": ms * 3},
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
	for fmt in ["native", "wasm"]:
		job = mk(fmt, 1500)
		resp = requests.post(f"{dt_url}/plan", json={"job": job, "strategy": "cvar", "dry_run": True}, timeout=20)
		resp.raise_for_status()
		print(json.dumps(resp.json()))


if __name__ == "__main__":
	run("http://localhost:8080")





