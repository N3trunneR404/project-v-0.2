#!/usr/bin/env python3
"""Continuous job generator for exercising the Fabric DT scheduler.

Example:
    python tools/load_generator.py \
        --jobs jobs/jobs_large_scale.yaml \
        --strategy resilient \
        --concurrency 12 \
        --interval 6

This script keeps posting jobs to the DT API so that pods remain active long
enough to observe migrations, chaos responses, and utilisation changes.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
import yaml


def load_jobs(path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
        return [item for item in jobs if isinstance(item, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Fabric DT load generator")
    parser.add_argument("--jobs", default="jobs/jobs_large_scale.yaml", type=Path)
    parser.add_argument("--strategy", default="resilient")
    parser.add_argument("--url", default="http://localhost:8080/plan")
    parser.add_argument("--concurrency", type=int, default=10, help="approximate in-flight plans")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between submissions")
    parser.add_argument("--dry-run", action="store_true", help="submit dry_run plans")
    args = parser.parse_args()

    jobs = load_jobs(args.jobs)
    if not jobs:
        raise SystemExit(f"No jobs found in {args.jobs}")

    session = requests.Session()
    inflight = 0
    counter = 0

    while True:
        try:
            if inflight < args.concurrency:
                job = json.loads(json.dumps(random.choice(jobs)))  # deep copy
                counter += 1
                response = session.post(
                    args.url,
                    json={"job": job, "strategy": args.strategy, "dry_run": args.dry_run},
                    timeout=20,
                )
                data = response.json()
                if data.get("ok"):
                    inflight += 1
                    plan = data["data"]
                    print(
                        f"[{time.strftime('%H:%M:%S')}] job #{counter} -> {job.get('id')} | "
                        f"lat={plan.get('latency_ms')}ms res={plan.get('resilience_score')} inflight={inflight}"
                    )
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] plan failed: {data.get('error')}")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] inflight limit reached ({inflight}), waiting…")
            time.sleep(max(0.5, args.interval))
            if inflight > 0:
                inflight -= 1  # optimistic decay; executor updates could be added later
        except KeyboardInterrupt:
            print("Stopping load generator")
            break
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] error: {exc}")
            time.sleep(5)


if __name__ == "__main__":
    main()
