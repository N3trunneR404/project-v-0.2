from __future__ import annotations

import json
import subprocess
import statistics as stats


def run_once(seconds: int = 2) -> float:
	cmd = ["stress-ng", "--cpu", "1", "--timeout", f"{seconds}s", "--metrics-brief"]
	subprocess.run(cmd, check=True, capture_output=True)
	return float(seconds * 1000)


def main():
	samples = [run_once(1), run_once(2), run_once(3)]
	out = {
		"ms": samples,
		"mean_ms": stats.mean(samples),
		"host": "local",
	}
	print(json.dumps(out, indent=2))


if __name__ == "__main__":
	main()





