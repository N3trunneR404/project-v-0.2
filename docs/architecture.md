# Digital Twin Controller Architecture

## 1. Overview
- Fabric: single-machine Kubernetes (k3d/kind) with KWOK-simulated nodes, Chaos Mesh for fault injection, `tc netem` for link shaping.
- Controller: Digital Twin (DT) process that mirrors cluster state, evaluates candidate plans in virtual time, then issues actions via Kubernetes APIs.
- Scope: Heuristic policies only (greedy latency, resilient, CVaR risk-aware). Reinforcement learning remains a stub for future work.
- Validation: All experiments run locally using multi-arch containers (native/QEMU/Wasm) and reproducible YAML configs.

## 2. Component Breakdown
### 2.1 DT Core (`dt/`)
- `state.py`: in-memory model of nodes, links, jobs, telemetry; thread-safe updates.
- `des_simulator.py`: discrete-event engine modelling queueing, contention, and failure injection (default path for predictions).
- `predict.py`: wraps the DES engine, retaining a legacy heuristic path for regression comparisons. Overhead data is calibrated via host benchmarks.
- `policy/`: heuristic planners (`greedy`, `resilient`, `cvar`) returning per-stage placements. `rl_stub.py` preserves interfaces for later ML integration.
- `actuator.py`: wraps Kubernetes client (`CoreV1Api`) to cordon nodes and create worker pods following DT placement decisions.
- `api.py`: Flask REST API exposing `/plan`, `/observe`, `/snapshot` for orchestration and telemetry.

### 2.2 Kubernetes Integration
- Admission path: clients call `POST /plan` with job spec; optional dry-run for simulation-only analysis.
- Execution path: Actuator submits worker pods with node pinning and execution format labels.
- Optional scheduler plug-in: `k8s_plugins/scorer` implements a thin HTTP scorer (Kubernetes Scheduler Framework extender); future work may port this to Go for in-tree integration.

### 2.3 Worker Runtimes (`images/worker/`)
- `worker.sh`: stress-based workload harness with architecture-specific code paths.
- Dockerfiles: native amd64, QEMU arm64/riscv64, Wasm placeholder (for runtimes like wasmtime/Krustlet).
- Build strategy: Docker Buildx + `qemu-user-static` enables cross-ISA containers on the host.

### 2.4 Telemetry & Network Modeling
- Prometheus (assumed) provides CPU/memory metrics; DT expects fresh telemetry within 1 second.
- Network shaping: `sim/network/netem-daemonset.yaml` applies delay/loss per node; Chaos Mesh scenarios inject partitions/blackouts.
- Assumptions: fail-stop failures, trustworthy telemetry, no adversarial nodes.

## 3. Sense–Simulate–Act Loop
1. **Sense**: DT ingests Kubernetes watch events, telemetry, and chaos notifications via `/observe`.
2. **Simulate**: `PredictiveSimulator.score_plan()` evaluates candidate placements using calibrated compute/network models and architecture overhead factors.
3. **Act**: Selected plan is executed through Kubernetes API; shadow placements are recorded for fast failover.

## 4. Simulation Methodology
- **Cluster substrate**: `deploy/k3d-cluster.yaml` (1 server + 2 agents), KWOK config (`sim/kwok/cluster.yaml`) to scale logical nodes without heavy resource cost.
- **Heterogeneity**: NodeFeatureDiscovery labels (via `deploy/nfd-values.yaml`), multi-arch images, Wasm runtime class (`deploy/runtimeclass-wasm.yaml`).
- **Chaos & network**: Chaos Mesh workflows (`chaos/scenarios/*`), tc/netem daemonset.
- **Calibration utilities**: `tools/calibrate_compute.py` (compute/energy proxy) and `tools/calibrate_network.py` (baseline RTT sampling).
- **Workloads**: `tools/load_generator.py`, `experiments/v1–v6`, `tools/trace_replay.py` for trace-driven validation.
- **Metrics**: JSON/CSV outputs under `reports/`, aggregated via `tools/analyze_results.py`.

## 5. Constraints & Rationale
- **No Kubernetes core modifications**: maintaining a custom kube-scheduler/kubelet fork is unsustainable; official extension points (webhooks, scheduler plugins) are leveraged instead.
- **Cross-OS limitation**: Linux namespaces/eBPF features powering the controller are absent on Windows, Android, iOS; hence mobile/desktop OS integration stays future work.
- **Portability cost**: QEMU adds 25–80% CPU overhead for ISA translation; Wasm currently lacks rich syscall/GPU access—DT policies avoid cross-ISA execution when deadlines are tight.
- **Mobile deployment impractical**: phone kernels require bootloader unlock for eBPF/cgroup changes, violate warranty, and suffer from aggressive power management.

## 6. Future Enhancements (Stubs Only)
- Reinforcement learning & neural models: placeholder interface in `dt/policy/rl_stub.py`.
- GNN-based failure propagation and federated DT hierarchy can plug into `dt/state` and `predict` without redesign.
- Coded computation and risk-aware optimization may extend the policy library with additional heuristics.

## 7. Assumptions Summary
- Accurate node/perf data every ≤1 s.
- QEMU/Wasm overhead calibrated on host machine holds within ±15% for CPU-bound jobs.
- tc/netem-induced latency approximates physical RTT within ±10%.
- Chaos scenarios reflect fail-stop behaviour (no Byzantine faults).
- Jobs declare truthful resource needs and deadlines.

## 8. Repository Layout (Post-Cleanup)
- `dt/`: DT core modules and heuristic policies.
- `deploy/`, `sim/`, `chaos/`: single-machine cluster configuration.
- `images/worker/`: multi-arch worker images.
- `experiments/`: curated scripts (V1–V6) aligned with validation matrix.
- `tools/`: calibration, analysis, trace replay utilities.
- Legacy artifacts removed: old planner/k8s scripts, redundant experiments, massive node manifests, legacy docs, UI dashboards, and heavy RL implementations.

