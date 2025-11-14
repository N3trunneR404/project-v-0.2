# Digital Twin Controller (Sanitised Edition)

This repository hosts a lightweight digital-twin controller that plans workloads across a heterogeneous Kubernetes fabric. The focus is validating the **simulation methodology** and **controller concept** on a single machine—no legacy dashboard, planners, or RL agents are shipped in this branch.

Key concepts are documented in `docs/architecture.md`.

## Quick Start
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

# Launch the API (default :8080)
python app.py
```

Submit a job:
```bash
curl -X POST http://127.0.0.1:8080/plan \
     -H "Content-Type: application/json" \
     -d @jobs/jobs_10.yaml
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
- **Cluster substrate:** `k3d` (or `kind`) + KWOK for inexpensive node fan-out.
- **Heterogeneity:** Docker Buildx multi-arch images, QEMU user emulation, optional Wasm runtime.
- **Network & chaos:** `sim/network/netem-daemonset.yaml` and `chaos/scenarios/*.yaml`.
- **Policies:** Greedy latency, resilient (shadow plans), risk-aware CVaR. RL is intentionally stubbed.
- **Predictor:** Discrete-event simulation (queueing, contention, optional failures) runs by default; the legacy heuristic remains available for regression tests.

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
