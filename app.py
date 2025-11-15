from __future__ import annotations

import os
import logging
from pathlib import Path

from dt.api import create_app
from dt.state import DTState, Node, HardwareSpec, NodeRuntime, KubernetesInfo, ClusterInfo
from dt.cluster_manager import ClusterManager

logger = logging.getLogger(__name__)


def seed_state(state: DTState) -> None:
	"""Seed the state with test nodes. Safe to call multiple times."""
	# Clear existing nodes first (in case this is called multiple times)
	if hasattr(state, 'nodes_by_name'):
		state.nodes_by_name.clear()
	elif hasattr(state, '_nodes'):
		state._nodes.clear()
	
	# Minimal seed of a few nodes for local testing
	for i in range(3):
		hw = HardwareSpec(cpu_cores=4, base_ghz=3.5, memory_gb=8, arch="amd64", tdp_w=95.0)
		rt = NodeRuntime(native_formats=["native"], emulation_support=["arm64", "riscv64"], wasm_support=True)
		k8s = KubernetesInfo(
			node_name=f"worker-{i}",
			labels={"dt.zone": "zone-a", "dt.cluster.name": "dc-core"},
			zone="zone-a",
			allocatable_cpu=4.0,
			allocatable_mem_gb=8.0
		)
		state.upsert_node(Node(name=f"worker-{i}", hardware=hw, runtime=rt, k8s=k8s))
	
	# Register default cluster if not already registered
	if "dc-core" not in state.clusters:
		cluster_info = ClusterInfo(
			name="dc-core",
			cluster_type="datacenter",
			resiliency_score=0.9,
			nodes=["worker-0", "worker-1", "worker-2"]
		)
		state.register_cluster(cluster_info)


def build_app():
	"""Build the Flask app with seeded state and cluster manager."""
	state = DTState()
	seed_state(state)
	
	# Initialize cluster manager with latency matrix
	latency_matrix_path = os.getenv(
		"LATENCY_MATRIX_PATH",
		str(Path(__file__).parent / "deploy" / "latency-matrix.yaml")
	)
	
	cluster_manager = None
	if os.path.exists(latency_matrix_path):
		try:
			cluster_manager = ClusterManager(latency_matrix_path=latency_matrix_path)
			logger.info(f"Initialized cluster manager with latency matrix: {latency_matrix_path}")
		except Exception as e:
			logger.warning(f"Failed to initialize cluster manager: {e}")
			cluster_manager = None
	else:
		logger.info("Latency matrix not found, running in single-cluster mode")
	
	return create_app(state, cluster_manager=cluster_manager)


# Build app at module level (for gunicorn)
app = build_app()

# Ensure state is seeded when module is imported (for worker processes)
# This will be called in each worker after forking
if hasattr(app, 'config'):
	state = app.config.get('dt_state')
	if state:
		# Verify nodes exist, re-seed if needed
		nodes = state.list_nodes()
		if not nodes:
			seed_state(state)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8080)





