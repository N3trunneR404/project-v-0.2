from __future__ import annotations

import time
import requests


def run(dt_url: str, jobs: int = 200) -> None:
	start = time.time()
	for i in range(jobs):
		j = {
			"apiVersion": "fabric.dt/v1",
			"kind": "Job",
			"metadata": {"name": f"scale-{i}", "deadline_ms": 1500},
			"spec": {"stages": [{"id": "s1", "compute": {"cpu": 1, "mem_gb": 1, "duration_ms": 800}, "constraints": {"arch": ["amd64"], "formats": ["native"]}}]},
		}
		requests.post(f"{dt_url}/plan", json={"job": j, "strategy": "greedy", "dry_run": True}, timeout=15)
	elapsed = (time.time() - start) * 1000.0
	print(f"jobs={jobs} total_ms={elapsed:.1f} avg_ms_per_job={elapsed/jobs:.1f}")


if __name__ == "__main__":
	run("http://localhost:8080", 200)





