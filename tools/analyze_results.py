from __future__ import annotations

import json
from pathlib import Path
import statistics as stats


def summarize_reports(path: str) -> None:
	files = list(Path(path).glob("*.json"))
	if not files:
		print("No JSON reports found")
		return
	latencies = []
	violations = 0
	for f in files:
		try:
			data = json.loads(f.read_text())
			lat = float(data.get("predicted_latency_ms", 0.0))
			latencies.append(lat)
			violations += int(data.get("violations", 0))
		except Exception:
			continue
	if latencies:
		print(f"count={len(latencies)} mean={stats.mean(latencies):.1f} p95={percentile(latencies,95):.1f} violations={violations}")


def percentile(values, p):
	values = sorted(values)
	k = (len(values)-1) * p/100.0
	f = int(k)
	c = min(f+1, len(values)-1)
	if f == c:
		return values[int(k)]
	return values[f] + (values[c] - values[f]) * (k - f)


def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--reports", default="reports/")
	args = parser.parse_args()
	summarize_reports(args.reports)


if __name__ == "__main__":
	main()

#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

REPORTS = Path("reports/experiments")
SUMMARY_FILE = REPORTS / "all_experiments_summary.json"
PLAN_LOG = Path("logs/plan_assignments.log")

def load_summary():
    data = json.loads(SUMMARY_FILE.read_text())
    return data

def plot_policy_comparison(summary):
    exp = summary["experiment_1"]["summary"]
    df = pd.DataFrame.from_dict(exp, orient="index")
    df = df.rename(columns={
        "mean_latency_ms": "Mean Latency (ms)",
        "p95_latency_ms": "p95 Latency (ms)",
        "mean_energy_kj": "Mean Energy (kJ)",
        "sla_violation_rate": "SLA Viol. Rate",
        "mean_resilience": "Mean Resilience"
    })
    plt.figure(figsize=(8,5))
    df["Mean Latency (ms)"].plot(kind="bar", color="#3277d5")
    plt.ylabel("Milliseconds")
    plt.title("Policy Comparison – Mean Latency")
    plt.tight_layout()
    plt.savefig(REPORTS / "policy_latency.png", dpi=300)

    plt.figure(figsize=(8,5))
    (df["Mean Energy (kJ)"]*1000).plot(kind="bar", color="#8e44ad")
    plt.ylabel("Energy (J)")
    plt.title("Policy Comparison – Mean Energy")
    plt.tight_layout()
    plt.savefig(REPORTS / "policy_energy.png", dpi=300)

    plt.figure(figsize=(8,5))
    df["Mean Resilience"].plot(kind="bar", color="#27ae60")
    plt.ylabel("Resilience")
    plt.ylim(0,1)
    plt.title("Policy Comparison – Resilience")
    plt.tight_layout()
    plt.savefig(REPORTS / "policy_resilience.png", dpi=300)

    plt.figure(figsize=(8,5))
    (df["SLA Viol. Rate"]*100).plot(kind="bar", color="#e74c3c")
    plt.ylabel("Violation Rate (%)")
    plt.title("Policy Comparison – SLA Violations")
    plt.tight_layout()
    plt.savefig(REPORTS / "policy_sla.png", dpi=300)

def plot_chaos(summary):
    exp = summary["experiment_2"]
    df = pd.DataFrame(exp.get("series", []))
    if df.empty:
        return
    plt.figure(figsize=(8,5))
    sns.barplot(x="policy", y="mean_recovery_ms", data=df, color="#f39c12")
    plt.ylabel("Recovery Time (ms)")
    plt.xlabel("Policy")
    plt.title("Chaos Resilience – Recovery Time")
    plt.tight_layout()
    plt.savefig(REPORTS / "chaos_recovery.png", dpi=300)

def plot_utilisation_from_log():
    if not PLAN_LOG.exists():
        return
    entries = [json.loads(line) for line in PLAN_LOG.read_text().strip().splitlines()]
    if not entries:
        return
    df = pd.DataFrame(entries)
    df["ts"] = pd.to_datetime(df["ts"])
    df["resilience"] = df["resilience"].fillna(0)
    plt.figure(figsize=(10,4))
    sns.lineplot(x="ts", y="resilience", data=df, hue="strategy", marker="o")
    plt.ylabel("Resilience Score")
    plt.xlabel("Time")
    plt.title("Resilience over time (from plan log)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(REPORTS / "resilience_timeline.png", dpi=300)

    df["assigned_nodes"] = df["assignments"].apply(lambda a: len(a) if isinstance(a, dict) else 0)
    plt.figure(figsize=(10,4))
    sns.lineplot(x="ts", y="assigned_nodes", data=df, hue="strategy", marker="o")
    plt.ylabel("Stages assigned")
    plt.xlabel("Time")
    plt.title("Stages per plan (live log)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(REPORTS / "assignment_timeline.png", dpi=300)

def main():
    summary = load_summary()
    plot_policy_comparison(summary)
    plot_chaos(summary)
    plot_utilisation_from_log()
    print("Plots written to", REPORTS.resolve())

if __name__ == "__main__":
    main()
