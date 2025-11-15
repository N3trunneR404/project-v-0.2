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

    try:
        existing_nodes = {node.name for node in state.list_nodes()}
    except Exception:
        existing_nodes = set()

    # Minimal seed of a few nodes for local testing
    for i in range(3):
        node_name = f"worker-{i}"
        hw = HardwareSpec(cpu_cores=4, base_ghz=3.5, memory_gb=8, arch="amd64", tdp_w=95.0)
        rt = NodeRuntime(native_formats=["native"], emulation_support=["arm64", "riscv64"], wasm_support=True)
        k8s = KubernetesInfo(
            node_name=node_name,
            labels={"dt.zone": "zone-a", "dt.cluster.name": "dc-core"},
            zone="zone-a",
            allocatable_cpu=4.0,
            allocatable_mem_gb=8.0,
        )
        node = Node(name=node_name, hardware=hw, runtime=rt, k8s=k8s)

        if node_name in existing_nodes:
            # Update existing node in place
            try:
                state.upsert_node(node)
            except AttributeError:
                if hasattr(state, "_nodes"):
                    state._nodes[node_name] = node
        else:
            state.upsert_node(node)

    # Register default cluster if not already registered
    cluster_info = ClusterInfo(
        name="dc-core",
        cluster_type="datacenter",
        resiliency_score=0.9,
        nodes=["worker-0", "worker-1", "worker-2"],
    )

    try:
        clusters = getattr(state, "clusters")
    except AttributeError:
        clusters = {}
        setattr(state, "clusters", clusters)

    if isinstance(clusters, dict) and "dc-core" in clusters:
        return

    if hasattr(state, "register_cluster"):
        state.register_cluster(cluster_info)
    else:
        clusters[cluster_info.name] = cluster_info


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





