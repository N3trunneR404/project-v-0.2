from __future__ import annotations

"""
Utility helpers for measuring baseline network latency/bandwidth on the host.

Usage:
    python tools/calibrate_network.py --interface eth0 --duration 5

Requires `ping` and (optionally) `iperf3` to be installed locally. The script
records a mean/95th percentile latency sample that can be plugged into the DT
simulator configuration.
"""

import argparse
import json
import statistics
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class PingStats:
	host: str
	mean_ms: float
	p95_ms: float


def run_ping(host: str, count: int = 20) -> PingStats:
	cmd = ["ping", "-c", str(count), host]
	output = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
	latencies: List[float] = []
	for line in output.splitlines():
		if "time=" in line:
			try:
				lat = float(line.split("time=")[1].split()[0])
				latencies.append(lat)
			except Exception:
				continue
	if not latencies:
		raise RuntimeError("ping produced no latency samples")
	return PingStats(host=host, mean_ms=statistics.mean(latencies), p95_ms=percentile(latencies, 95))


def percentile(values: List[float], p: float) -> float:
	values = sorted(values)
	if not values:
		return 0.0
	k = (len(values) - 1) * p / 100.0
	f = int(k)
	c = min(f + 1, len(values) - 1)
	if f == c:
		return values[int(k)]
	return values[f] + (values[c] - values[f]) * (k - f)


def main() -> None:
	parser = argparse.ArgumentParser(description="Calibrate baseline network latency")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--count", type=int, default=20)
	args = parser.parse_args()

	stats = run_ping(args.host, args.count)
	print(json.dumps({"host": stats.host, "mean_ms": stats.mean_ms, "p95_ms": stats.p95_ms}, indent=2))


if __name__ == "__main__":
	main()

