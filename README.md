# Digital Twin Controller (Multi-Cluster Edition)

This repository hosts a digital-twin controller that plans workloads across a heterogeneous multi-cluster Kubernetes fabric. The DT mirrors real infrastructure, collects actual telemetry, performs origin-aware placement, injects failures, and verifies predictions against observed metrics.

**Key Features:**
- **Multi-cluster architecture**: 6-7 k3d clusters representing diverse device pools (datacenter, mining, lab, gaming, PAN, edge)
- **Real telemetry collection**: Kubernetes Metrics API + cAdvisor + watch API for pod/node lifecycle
- **Origin-aware placement**: Policies consider job origin cluster/node when computing placement latency
- **Failure injection**: Time-based and event-driven failures weighted by cluster resiliency scores
- **Prediction verification**: Compare predicted vs observed metrics with configurable thresholds
- **Resource scaling**: 1:100 default scaling (configurable) for large-scale simulation on single machine

## Documentation

- **Comprehensive Report**: See `FINAL_REPORT.md` for detailed problem statement, methodology, implementation, results, and analysis
- **Architecture Details**: See `docs/architecture.md` for technical architecture and component breakdown
- **Worker Images**: See `images/worker/README.md` for multi-architecture container build instructions

## Quick Start

### Setup Multi-Cluster Environment
```bash
# Create 6-7 k3d clusters with metrics-server
./deploy/multi-cluster-setup.sh

# Or for single cluster (legacy mode)
k3d cluster create fabric-dt --config deploy/k3d-cluster.yaml
```

### Launch the API
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch the API (default :8080)
python app.py
```

### Submit a Job with Origin Context
```bash
curl -X POST http://127.0.0.1:8080/plan \
     -H "Content-Type: application/json" \
     -d '{
       "job": {
         "metadata": {
           "name": "test-job",
           "deadline_ms": 10000,
           "origin": {
             "cluster": "edge-microdc",
             "node": "edge-node-01"
           }
         },
         "spec": {
           "stages": [{
             "id": "s1",
             "compute": {"cpu": 2, "mem_gb": 1, "duration_ms": 2000},
             "constraints": {"arch": ["amd64"], "formats": ["native"]}
           }]
         }
       },
       "strategy": "greedy"
     }'
```

### Verify Prediction Accuracy
```bash
# After job execution, check verification results
curl http://127.0.0.1:8080/plan/<plan_id>/verify
```

## Repository Layout
```
.
├── dt/              # DT core (state, predictor, policies, actuator, API)
├── deploy/          # k3d cluster, runtime classes, NFD configuration
├── sim/             # KWOK spec, netem daemonset, chaos scaffolding
├── chaos/           # Chaos Mesh scenarios (zone blackout, partition)
├── experiments/     # Validation scripts (V1–V6)
├── images/worker/   # Multi-arch worker images + execution script
├── k8s_executor/    # Pod generator (DT plan -> PodSpec)
├── k8s_plugins/     # Optional scheduler scorer stub
├── tools/           # Calibration, analysis, trace replay helpers
├── docs/            # Architecture and assumption notes
├── jobs/            # Sample job descriptors
└── requirements.txt # Runtime dependencies
```

Removed components: legacy planner, dashboard UI, Monte-Carlo runners, massive node catalogs, historical docs, RL implementations, and redundant experiments. They can be reintroduced later if required.

## Simulation Stack (High Level)
- **Multi-cluster substrate:** 6-7 k3d clusters + KWOK for inexpensive node fan-out. Inter-cluster latency matrix for origin-aware placement.
- **Heterogeneity:** Docker Buildx multi-arch images, QEMU user emulation, optional Wasm runtime.
- **Network & chaos:** Inter-cluster latency matrix, `sim/network/netem-daemonset.yaml`, and `chaos/scenarios/*.yaml`. Failure injection system (time-based and event-driven).
- **Policies:** Greedy latency, resilient (shadow plans), risk-aware CVaR, all with origin-aware placement scoring. RL is intentionally stubbed.
- **Predictor:** Discrete-event simulation (queueing, contention, optional failures) runs by default with resource scaling support; the legacy heuristic remains available for regression tests.
- **Telemetry:** Real metrics collection from Kubernetes Metrics API, cAdvisor, and watch API. Continuous monitoring of pod/node lifecycle.
- **Verification:** Prediction accuracy validation comparing predicted vs observed metrics (latency ±10%, energy ±20% thresholds).

For detailed diagrams, assumptions, and experiment definitions, see `docs/architecture.md`.

## Housekeeping
- Lint: `ruff check .`
- Format: `black .`
- Experiments: run individual scripts under `experiments/` or `python -m experiments.run_suite`; outputs land in `reports/`.

## Limitations
- No smart/context-aware (hardware-level) scheduler yet—kernel hooks and cross-OS agents remain future work.
- Mobile deployment is unsupported (requires bootloader unlock/kernel patches).
- Cross-ISA execution incurs measured QEMU/Wasm overhead; policies avoid it unless slack allows.

Future enhancements should plug stubs in `dt/policy/rl_stub.py` and extend the policy library without reintroducing the removed legacy stack.
