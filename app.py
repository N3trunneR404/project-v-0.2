from __future__ import annotations

import os
import logging
from pathlib import Path

from dt.api import create_app
from dt.state import (
    DTState,
    Node,
    HardwareSpec,
    NodeRuntime,
    KubernetesInfo,
    ClusterInfo,
)
from dt.cluster_manager import ClusterManager

logger = logging.getLogger(__name__)


def seed_state(state: DTState) -> None:
    """Seed the state with test nodes. Safe to call multiple times."""

    try:
        existing_nodes = {node.name for node in state.list_nodes()}
    except Exception:
        existing_nodes = set()

    cluster_specs = [
        {
            "info": ClusterInfo(
                name="dc-core",
                cluster_type="datacenter",
                resiliency_score=0.95,
                network_cidr="172.30.10.0/24",
                pod_cidr="10.10.0.0/16",
                service_cidr="10.110.0.0/16",
            ),
            "nodes": [
                {
                    "name": "dc-core-master-0",
                    "hardware": dict(cpu_cores=32, base_ghz=3.2, memory_gb=256, tdp_w=205.0,
                                      memory_mhz=5600, memory_type="DDR5", accelerators=["gpu:nvidia-a100"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-central1-a",
                        "dt.cluster.name": "dc-core",
                        "dt.cluster.type": "datacenter",
                        "dt.hardware.tdp": "205",
                        "dt.hardware.memory_mhz": "5600",
                        "dt.hardware.accelerators": "gpu:nvidia-a100",
                    },
                },
                {
                    "name": "dc-core-worker-1",
                    "hardware": dict(cpu_cores=48, base_ghz=3.0, memory_gb=384, tdp_w=240.0,
                                      memory_mhz=5200, memory_type="DDR5", accelerators=["gpu:nvidia-h100", "fpga:xilinx-u250"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-central1-a",
                        "dt.cluster.name": "dc-core",
                        "dt.cluster.type": "datacenter",
                        "dt.hardware.tdp": "240",
                        "dt.hardware.memory_mhz": "5200",
                        "dt.hardware.accelerators": "gpu:nvidia-h100,fpga:xilinx-u250",
                    },
                },
                {
                    "name": "dc-core-worker-2",
                    "hardware": dict(cpu_cores=24, base_ghz=3.4, memory_gb=192, tdp_w=180.0,
                                      memory_mhz=4800, memory_type="DDR5", accelerators=["gpu:nvidia-rtx6000"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64", "riscv64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-central1-b",
                        "dt.cluster.name": "dc-core",
                        "dt.cluster.type": "datacenter",
                        "dt.hardware.tdp": "180",
                        "dt.hardware.memory_mhz": "4800",
                        "dt.hardware.accelerators": "gpu:nvidia-rtx6000",
                    },
                },
            ],
        },
        {
            "info": ClusterInfo(
                name="prosumer-mining",
                cluster_type="mining",
                resiliency_score=0.7,
                network_cidr="172.30.11.0/24",
                pod_cidr="10.11.0.0/16",
                service_cidr="10.111.0.0/16",
            ),
            "nodes": [
                {
                    "name": "prosumer-mining-master-0",
                    "hardware": dict(cpu_cores=16, base_ghz=3.8, memory_gb=64, tdp_w=125.0,
                                      memory_mhz=4400, memory_type="DDR5", accelerators=["gpu:nvidia-rtx3090"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=False),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-central1-c",
                        "dt.cluster.name": "prosumer-mining",
                        "dt.cluster.type": "mining",
                        "dt.hardware.tdp": "125",
                        "dt.hardware.memory_mhz": "4400",
                        "dt.hardware.accelerators": "gpu:nvidia-rtx3090",
                    },
                },
                {
                    "name": "prosumer-mining-worker-1",
                    "hardware": dict(cpu_cores=12, base_ghz=3.6, memory_gb=48, tdp_w=105.0,
                                      memory_mhz=4266, memory_type="DDR4", accelerators=["gpu:nvidia-rtx3080"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=False),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-central1-c",
                        "dt.cluster.name": "prosumer-mining",
                        "dt.cluster.type": "mining",
                        "dt.hardware.tdp": "105",
                        "dt.hardware.memory_mhz": "4266",
                        "dt.hardware.accelerators": "gpu:nvidia-rtx3080",
                    },
                },
                {
                    "name": "prosumer-mining-worker-2",
                    "hardware": dict(cpu_cores=10, base_ghz=3.5, memory_gb=32, tdp_w=95.0,
                                      memory_mhz=3600, memory_type="DDR4", accelerators=["gpu:nvidia-rtx3070"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=False),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-central1-d",
                        "dt.cluster.name": "prosumer-mining",
                        "dt.cluster.type": "mining",
                        "dt.hardware.tdp": "95",
                        "dt.hardware.memory_mhz": "3600",
                        "dt.hardware.accelerators": "gpu:nvidia-rtx3070",
                    },
                },
            ],
        },
        {
            "info": ClusterInfo(
                name="campus-lab",
                cluster_type="lab",
                resiliency_score=0.85,
                network_cidr="172.30.12.0/24",
                pod_cidr="10.12.0.0/16",
                service_cidr="10.112.0.0/16",
            ),
            "nodes": [
                {
                    "name": "campus-lab-master-0",
                    "hardware": dict(cpu_cores=12, base_ghz=3.2, memory_gb=64, tdp_w=95.0,
                                      memory_mhz=3600, memory_type="DDR4", accelerators=["gpu:nvidia-tesla-t4"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64", "riscv64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-east1-a",
                        "dt.cluster.name": "campus-lab",
                        "dt.cluster.type": "lab",
                        "dt.hardware.tdp": "95",
                        "dt.hardware.memory_mhz": "3600",
                        "dt.hardware.accelerators": "gpu:nvidia-tesla-t4",
                    },
                },
                {
                    "name": "campus-lab-worker-1",
                    "hardware": dict(cpu_cores=8, base_ghz=3.1, memory_gb=48, tdp_w=80.0,
                                      memory_mhz=3200, memory_type="DDR4", accelerators=[]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64", "riscv64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-east1-b",
                        "dt.cluster.name": "campus-lab",
                        "dt.cluster.type": "lab",
                        "dt.hardware.tdp": "80",
                        "dt.hardware.memory_mhz": "3200",
                        "dt.hardware.accelerators": "none",
                    },
                },
                {
                    "name": "campus-lab-worker-2",
                    "hardware": dict(cpu_cores=8, base_ghz=3.0, memory_gb=32, tdp_w=75.0,
                                      memory_mhz=3200, memory_type="DDR4", accelerators=[]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64", "riscv64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "us-east1-b",
                        "dt.cluster.name": "campus-lab",
                        "dt.cluster.type": "lab",
                        "dt.hardware.tdp": "75",
                        "dt.hardware.memory_mhz": "3200",
                        "dt.hardware.accelerators": "none",
                    },
                },
            ],
        },
        {
            "info": ClusterInfo(
                name="edge-microdc",
                cluster_type="edge",
                resiliency_score=0.8,
                network_cidr="172.30.13.0/24",
                pod_cidr="10.13.0.0/16",
                service_cidr="10.113.0.0/16",
            ),
            "nodes": [
                {
                    "name": "edge-microdc-master-0",
                    "hardware": dict(cpu_cores=16, base_ghz=2.8, memory_gb=64, tdp_w=95.0,
                                      memory_mhz=3200, memory_type="DDR4", accelerators=["gpu:nvidia-l4"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "europe-west1-a",
                        "dt.cluster.name": "edge-microdc",
                        "dt.cluster.type": "edge",
                        "dt.hardware.tdp": "95",
                        "dt.hardware.memory_mhz": "3200",
                        "dt.hardware.accelerators": "gpu:nvidia-l4",
                    },
                },
                {
                    "name": "edge-microdc-worker-1",
                    "hardware": dict(cpu_cores=12, base_ghz=2.6, memory_gb=48, tdp_w=80.0,
                                      memory_mhz=3000, memory_type="DDR4", accelerators=["npu:habana-gaudi"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "europe-west1-b",
                        "dt.cluster.name": "edge-microdc",
                        "dt.cluster.type": "edge",
                        "dt.hardware.tdp": "80",
                        "dt.hardware.memory_mhz": "3000",
                        "dt.hardware.accelerators": "npu:habana-gaudi",
                    },
                },
                {
                    "name": "edge-microdc-worker-2",
                    "hardware": dict(cpu_cores=8, base_ghz=2.4, memory_gb=32, tdp_w=65.0,
                                      memory_mhz=2800, memory_type="DDR4", accelerators=[]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "europe-west1-c",
                        "dt.cluster.name": "edge-microdc",
                        "dt.cluster.type": "edge",
                        "dt.hardware.tdp": "65",
                        "dt.hardware.memory_mhz": "2800",
                        "dt.hardware.accelerators": "none",
                    },
                },
                {
                    "name": "edge-microdc-worker-3",
                    "hardware": dict(cpu_cores=6, base_ghz=2.2, memory_gb=24, tdp_w=45.0,
                                      memory_mhz=2666, memory_type="DDR4", accelerators=[]),
                    "runtime": dict(native_formats=["native"], emulation_support=["arm64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "europe-west1-c",
                        "dt.cluster.name": "edge-microdc",
                        "dt.cluster.type": "edge",
                        "dt.hardware.tdp": "45",
                        "dt.hardware.memory_mhz": "2666",
                        "dt.hardware.accelerators": "none",
                    },
                },
            ],
        },
        {
            "info": ClusterInfo(
                name="phone-pan-1",
                cluster_type="pan",
                resiliency_score=0.6,
                network_cidr="172.30.14.0/24",
                pod_cidr="10.14.0.0/16",
                service_cidr="10.114.0.0/16",
            ),
            "nodes": [
                {
                    "name": "phone-pan-1-master-0",
                    "hardware": dict(cpu_cores=8, base_ghz=2.0, memory_gb=12, tdp_w=25.0,
                                      memory_mhz=2133, memory_type="LPDDR5", accelerators=["npu:qualcomm-hvx"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["amd64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "asia-south1-a",
                        "dt.cluster.name": "phone-pan-1",
                        "dt.cluster.type": "pan",
                        "dt.hardware.tdp": "25",
                        "dt.hardware.memory_mhz": "2133",
                        "dt.hardware.accelerators": "npu:qualcomm-hvx",
                    },
                },
                {
                    "name": "phone-pan-1-worker-1",
                    "hardware": dict(cpu_cores=6, base_ghz=1.8, memory_gb=8, tdp_w=18.0,
                                      memory_mhz=2133, memory_type="LPDDR5", accelerators=["gpu:adreno-730"]),
                    "runtime": dict(native_formats=["native"], emulation_support=["amd64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "asia-south1-b",
                        "dt.cluster.name": "phone-pan-1",
                        "dt.cluster.type": "pan",
                        "dt.hardware.tdp": "18",
                        "dt.hardware.memory_mhz": "2133",
                        "dt.hardware.accelerators": "gpu:adreno-730",
                    },
                },
                {
                    "name": "phone-pan-1-worker-2",
                    "hardware": dict(cpu_cores=4, base_ghz=1.6, memory_gb=6, tdp_w=12.0,
                                      memory_mhz=1866, memory_type="LPDDR4", accelerators=[]),
                    "runtime": dict(native_formats=["native"], emulation_support=["amd64"], wasm_support=True),
                    "labels": {
                        "topology.kubernetes.io/zone": "asia-south1-c",
                        "dt.cluster.name": "phone-pan-1",
                        "dt.cluster.type": "pan",
                        "dt.hardware.tdp": "12",
                        "dt.hardware.memory_mhz": "1866",
                        "dt.hardware.accelerators": "none",
                    },
                },
            ],
        },
    ]

    for spec in cluster_specs:
        cluster_info = spec["info"]
        if hasattr(state, "register_cluster"):
            state.register_cluster(cluster_info)
        for node_spec in spec["nodes"]:
            node_name = node_spec["name"]
            hw = HardwareSpec(
                cpu_cores=node_spec["hardware"]["cpu_cores"],
                base_ghz=node_spec["hardware"]["base_ghz"],
                memory_gb=node_spec["hardware"]["memory_gb"],
                arch="amd64",
                tdp_w=node_spec["hardware"]["tdp_w"],
                memory_mhz=node_spec["hardware"].get("memory_mhz"),
                memory_type=node_spec["hardware"].get("memory_type"),
                accelerators=node_spec["hardware"].get("accelerators", []),
            )
            rt = NodeRuntime(
                native_formats=node_spec["runtime"].get("native_formats", ["native"]),
                emulation_support=node_spec["runtime"].get("emulation_support", []),
                wasm_support=node_spec["runtime"].get("wasm_support", False),
            )
            labels = dict(node_spec.get("labels", {}))
            labels.setdefault("dt.cluster.name", cluster_info.name)
            labels.setdefault("dt.cluster.type", cluster_info.cluster_type)
            k8s = KubernetesInfo(
                node_name=node_name,
                labels=labels,
                allocatable_cpu=float(hw.cpu_cores),
                allocatable_mem_gb=float(hw.memory_gb),
                zone=labels.get("topology.kubernetes.io/zone"),
            )
            node = Node(name=node_name, hardware=hw, runtime=rt, k8s=k8s)

            if node_name in existing_nodes:
                state.upsert_node(node)
            else:
                state.upsert_node(node)

    # Force rebuild of virtual topology if the implementation supports it
    try:
        state.describe_virtual_topology()
    except Exception:
        pass


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





