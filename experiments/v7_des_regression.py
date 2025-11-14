"""
Experiment V7: Compare discrete-event simulation (DES) vs. legacy heuristic
predictions for a simple workload. This script creates an in-memory DT state
with a few heterogeneous nodes, assigns a two-stage job, and prints the
predicted metrics from both engines so regressions are easy to spot.
"""

from __future__ import annotations

from dataclasses import dataclass

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from dt.predict import PredictiveSimulator
from dt.state import (
	DTState,
	HardwareSpec,
	Job,
	JobStage,
	KubernetesInfo,
	Node,
	NodeRuntime,
	PlacementDecision,
	StageCompute,
	StageConstraints,
)


def build_state() -> DTState:
	state = DTState()
	nodes = [
		Node(
			name="hpc-001",
			hardware=HardwareSpec(cpu_cores=16, base_ghz=3.2, memory_gb=64, gpu_vram_gb=16, arch="amd64", tdp_w=150),
			runtime=NodeRuntime(native_formats=["native", "cuda"], wasm_support=True),
			k8s=KubernetesInfo(node_name="hpc-001", zone="datacenter", allocatable_cpu=16.0, allocatable_mem_gb=64.0),
		),
		Node(
			name="edge-002",
			hardware=HardwareSpec(cpu_cores=8, base_ghz=2.4, memory_gb=24, arch="arm64", tdp_w=85),
			runtime=NodeRuntime(native_formats=["native"], emulation_support=["amd64"], wasm_support=True),
			k8s=KubernetesInfo(node_name="edge-002", zone="edge"),
		),
	]
	for node in nodes:
		state.upsert_node(node)
	return state


def build_job() -> Job:
	stage_a = JobStage(
		id="ingest",
		compute=StageCompute(cpu=2, mem_gb=4, duration_ms=900, workload_type="io_bound"),
		constraints=StageConstraints(arch=["amd64"], formats=["native", "wasm"]),
		predecessor=None,
	)
	stage_b = JobStage(
		id="inference",
		compute=StageCompute(cpu=4, mem_gb=8, gpu_vram_gb=4, duration_ms=2200, workload_type="gpu_bound"),
		constraints=StageConstraints(arch=["amd64"], formats=["native", "cuda"]),
		predecessor="ingest",
	)
	return Job(name="sample-job", deadline_ms=4500, stages=[stage_a, stage_b])


def build_placements() -> dict[str, PlacementDecision]:
	return {
		"ingest": PlacementDecision(stage_id="ingest", node_name="edge-002", exec_format="qemu-amd64"),
		"inference": PlacementDecision(stage_id="inference", node_name="hpc-001", exec_format="native"),
	}


def main() -> None:
	state = build_state()
	job = build_job()
	placements = build_placements()

	simulator = PredictiveSimulator(state)
	des_metrics = simulator.score_plan(job, placements)
	legacy_metrics = simulator.score_plan_legacy(job, placements)

	print("=== DES vs. Legacy Prediction Comparison ===")
	print(f"Job name: {job.name}")
	print("\n-- DES (default) --")
	print(f"Latency (ms):         {des_metrics.latency_ms:.2f}")
	print(f"Energy (kWh):         {des_metrics.energy_kwh:.6f}")
	print(f"SLA violations:       {des_metrics.sla_violations}")
	print(f"Risk score:           {des_metrics.risk_score:.4f}")
	print(f"Completed stages:     {des_metrics.completed_stages}")
	print(f"Failed stages:        {des_metrics.failed_stages}")

	print("\n-- Legacy heuristic (for reference) --")
	print(f"Latency (ms):         {legacy_metrics.latency_ms:.2f}")
	print(f"Energy (kWh):         {legacy_metrics.energy_kwh:.6f}")
	print(f"SLA violations:       {legacy_metrics.sla_violations}")
	print(f"Risk score:           {legacy_metrics.risk_score:.4f}")

	print("\nDelta (DES - legacy):")
	print(f"Latency delta (ms):   {des_metrics.latency_ms - legacy_metrics.latency_ms:.2f}")
	print(f"Energy delta (kWh):   {des_metrics.energy_kwh - legacy_metrics.energy_kwh:.6f}")


if __name__ == "__main__":
	main()

