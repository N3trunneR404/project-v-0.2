from __future__ import annotations

from dt.api import create_app
from dt.state import DTState, Node, HardwareSpec, NodeRuntime, KubernetesInfo


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
		k8s = KubernetesInfo(node_name=f"worker-{i}", labels={"dt.zone": "zone-a"}, zone="zone-a", allocatable_cpu=4.0, allocatable_mem_gb=8.0)
		state.upsert_node(Node(name=f"worker-{i}", hardware=hw, runtime=rt, k8s=k8s))


def build_app():
	"""Build the Flask app with seeded state."""
	state = DTState()
	seed_state(state)
	return create_app(state)


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





