import os
import sys
from pathlib import Path

os.environ.setdefault("DT_AUTO_WATCHERS", "0")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from app import seed_state
from dt.state import (
    DTState,
    PlacementDecision,
    StageCompute,
    StageConstraints,
    JobStage,
    Job,
    JobOrigin,
)
from dt.predict import PredictiveSimulator
from dt.policy.greedy import GreedyLatencyPolicy
from dt.api import create_app


@pytest.fixture
def seeded_state():
    state = DTState(auto_start_watchers=False)
    seed_state(state)
    try:
        yield state
    finally:
        try:
            state.stop()
        except Exception:
            pass


def test_virtual_topology_assigns_unique_ids(seeded_state):
    nodes = seeded_state.list_nodes()
    assert nodes, "state should seed nodes"

    identity_pairs = set()
    cluster_ids = set()
    for node in nodes:
        assert node.network is not None
        pair = (node.network.cluster_id, node.network.node_id)
        assert pair not in identity_pairs
        identity_pairs.add(pair)
        cluster_ids.add(node.network.cluster_id)
        assert node.pool_key

    registered = {
        info.cluster_id for info in seeded_state.clusters.values() if info.cluster_id is not None
    }
    assert cluster_ids == registered

    for info in seeded_state.clusters.values():
        assert info.network_cidr
        assert info.pod_cidr
        assert info.service_cidr


def test_dt_policy_beats_baseline(seeded_state):
    sim = PredictiveSimulator(seeded_state)
    policy = GreedyLatencyPolicy(seeded_state, sim)

    job = Job(
        name="gpu-demo",
        deadline_ms=20000,
        stages=[
            JobStage(
                id="stage-1",
                compute=StageCompute(
                    cpu=12,
                    mem_gb=32,
                    duration_ms=5000,
                    gpu_vram_gb=16,
                    workload_type="gpu_bound",
                ),
                constraints=StageConstraints(arch=["amd64"], formats=["native"]),
                predecessor=None,
            )
        ],
        origin=JobOrigin(cluster="campus-lab", node="campus-lab-master-0"),
    )

    dt_plan = policy.place(job)
    assert dt_plan, "planner should find a placement"

    baseline = {}
    for stage in job.stages:
        for node in seeded_state.list_nodes():
            if stage.compute.gpu_vram_gb > 0 and node.hardware.gpu_vram_gb < stage.compute.gpu_vram_gb:
                continue
            if not node.available:
                continue
            baseline[stage.id] = PlacementDecision(
                stage_id=stage.id,
                node_name=node.name,
                exec_format="native",
            )
            break
    assert baseline, "baseline scheduler should pick at least one node"

    dt_metrics = sim.score_plan(job, dt_plan)
    baseline_metrics = sim.score_plan(job, baseline)

    assert dt_metrics.latency_ms < baseline_metrics.latency_ms


def test_plan_endpoint_returns_placements(seeded_state):
    app = create_app(seeded_state, cluster_manager=None)
    client = app.test_client()

    payload = {
        "job": {
            "metadata": {
                "name": "api-gpu",
                "deadline_ms": 25000,
                "origin": {"cluster": "campus-lab", "node": "campus-lab-master-0"},
            },
            "spec": {
                "stages": [
                    {
                        "id": "stage-1",
                        "compute": {
                            "cpu": 8,
                            "mem_gb": 16,
                            "duration_ms": 4000,
                            "gpu_vram_gb": 12,
                            "workload_type": "gpu_bound",
                        },
                        "constraints": {"arch": ["amd64"], "formats": ["native"]},
                    }
                ]
            },
        }
    }

    resp = client.post("/plan", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["placements"], "API should return placements"

    topo_resp = client.get("/topology/virtual")
    assert topo_resp.status_code == 200
    topology = topo_resp.get_json().get("virtual_topology")
    assert topology, "virtual topology endpoint should return data"
