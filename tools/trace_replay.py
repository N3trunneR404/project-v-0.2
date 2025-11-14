from __future__ import annotations

import csv
import time
import json
from pathlib import Path
from typing import Dict, Any
import requests


def load_trace(csv_path: str) -> list[dict[str, str]]:
	with open(csv_path, newline="") as f:
		return list(csv.DictReader(f))


def to_job(entry: dict[str, str]) -> Dict[str, Any]:
	name = f"job-{entry.get('job_id','x')}-{int(time.time())}"
	duration_ms = int(float(entry.get("duration_ms", "1000")))
	deadline_ms = max(duration_ms + 500, duration_ms * 2)
	job = {
		"apiVersion": "fabric.dt/v1",
		"kind": "Job",
		"metadata": {"name": name, "deadline_ms": deadline_ms},
		"spec": {
			"stages": [
				{
					"id": "s1",
					"compute": {"cpu": 1, "mem_gb": 1, "duration_ms": duration_ms},
					"constraints": {"arch": ["amd64"], "formats": ["native"]},
				}
			]
		},
	}
	return job


def submit(dt_url: str, job: Dict[str, Any]) -> Dict[str, Any]:
	resp = requests.post(f"{dt_url}/plan", json={"job": job, "strategy": "resilient", "dry_run": False}, timeout=30)
	resp.raise_for_status()
	return resp.json()


def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--trace", required=True)
	parser.add_argument("--dt-url", default="http://localhost:8080")
	args = parser.parse_args()

	entries = load_trace(args.trace)
	for e in entries:
		job = to_job(e)
		out = submit(args.dt_url, job)
		print(json.dumps(out))
		time.sleep(0.1)


if __name__ == "__main__":
	main()





