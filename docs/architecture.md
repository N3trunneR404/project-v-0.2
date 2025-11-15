# Digital Twin Controller Architecture

> **Note**: For a comprehensive academic report including problem statement, objectives, methodology, implementation details, results analysis, and conclusions, see `FINAL_REPORT.md` in the project root.

## 1. Overview
- Fabric: Multi-cluster Kubernetes setup (6-7 k3d clusters) representing diverse device pools (datacenter, mining, lab, gaming, PAN, edge), with KWOK-simulated nodes, Chaos Mesh for fault injection, `tc netem` for link shaping.
- Controller: Digital Twin (DT) process that mirrors multi-cluster state, collects real telemetry, evaluates candidate plans in virtual time with origin-aware placement, then issues actions via Kubernetes APIs across clusters.
- Scope: Heuristic policies (greedy latency, resilient, CVaR risk-aware) with origin-aware placement, real telemetry collection, failure injection, and prediction verification. Reinforcement learning remains a stub for future work.
- Validation: All experiments run locally using multi-arch containers (native/QEMU/Wasm) with resource scaling (1:100 default) and reproducible YAML configs.

## 2. Component Breakdown
### 2.1 DT Core (`dt/`)
- `state.py`: in-memory model of nodes, links, jobs, clusters, observed metrics, and job origin context; thread-safe updates.
- `cluster_manager.py`: manages connections to multiple Kubernetes clusters, maintains inter-cluster latency matrix, provides unified API for multi-cluster operations.
- `des_simulator.py`: discrete-event engine modelling queueing, contention, and failure injection (default path for predictions). Supports resource scaling.
- `predict.py`: wraps the DES engine, retaining a legacy heuristic path for regression comparisons. Overhead data is calibrated via host benchmarks.
- `scaling.py`: resource scaling framework (default 1:100) for simulation purposes, allowing large-scale workloads on single-machine setup.
- `policy/`: heuristic planners (`greedy`, `resilient`, `cvar`) with origin-aware placement scoring. Policies consider job origin cluster/node when computing placement latency.
- `actuator.py`: wraps Kubernetes clients for multiple clusters (`CoreV1Api` per cluster) to cordon nodes, create worker pods, and inject failures following DT placement decisions.
- `api.py`: Flask REST API exposing `/plan`, `/observe`, `/snapshot`, `/plan/<plan_id>/verify` for orchestration, telemetry, and prediction verification.
- `telemetry/collector.py`: collects real metrics from Kubernetes Metrics API, cAdvisor, and watch API for pod/node lifecycle events.
- `failures/`: failure event generator (time-based and event-driven) and resiliency scorer for cluster/node health assessment.
- `verification.py`: prediction verification module comparing predicted vs observed metrics with configurable delta thresholds.

### 2.2 Kubernetes Integration
- Multi-cluster setup: 6-7 k3d clusters (dc-core, prosumer-mining, campus-lab, gamer-pc, phone-pan-1, phone-pan-2, edge-microdc) each representing different device pool characteristics.
- Admission path: clients call `POST /plan` with job spec including origin context (cluster, node); optional dry-run for simulation-only analysis.
- Execution path: Actuator submits worker pods to the correct cluster based on placement decision, with node pinning, execution format labels, and scaled resource requirements.
- Telemetry: Metrics API + cAdvisor for CPU/memory metrics, watch API for pod/node lifecycle events, streaming into DT state.
- Failure injection: Time-based and event-driven failure generation (node_down, thermal_throttle, network_degradation, system_crash) weighted by cluster resiliency scores.
- Verification: Observed metrics collected from pod execution, compared against predictions with configurable thresholds (latency ±10%, energy ±20%).
- Optional scheduler plug-in: `k8s_plugins/scorer` implements a thin HTTP scorer (Kubernetes Scheduler Framework extender); future work may port this to Go for in-tree integration.

### 2.3 Worker Runtimes (`images/worker/`)
- `worker.sh`: stress-based workload harness with architecture-specific code paths. Consumes real CPU/memory resources scaled by resource_scale factor (default 1:100).
- Dockerfiles: native amd64, QEMU arm64/riscv64, Wasm placeholder (for runtimes like wasmtime/Krustlet).
- Build strategy: Docker Buildx + `qemu-user-static` enables cross-ISA containers on the host.
- Resource scaling: Simulated CPU/memory units are scaled down for real deployment (e.g., 100 simulated CPU = 1 real core).

### 2.4 Telemetry & Network Modeling
- Real telemetry collection: Kubernetes Metrics API + cAdvisor for CPU/memory metrics, watch API for pod/node lifecycle events. Metrics collected every 5 seconds (configurable).
- Network modeling: Inter-cluster latency matrix (`deploy/latency-matrix.yaml`) defines latencies between clusters. Network shaping: `sim/network/netem-daemonset.yaml` applies delay/loss per node; Chaos Mesh scenarios inject partitions/blackouts.
- Failure injection: Time-based failures (every 60s, weighted by resiliency) and event-driven failures (CPU > 80%, memory > 80%, queue length thresholds).
- Resiliency scoring: Cluster/node resiliency scores computed from historical failures, uptime, utilization, updated based on runtime observations.
- Prediction verification: Observed metrics (latency, CPU, memory, energy) compared against predictions with configurable thresholds, exported to CSV for analysis.
- Assumptions: fail-stop failures, trustworthy telemetry, no adversarial nodes.

## 3. Sense–Simulate–Act–Verify Loop
1. **Sense**: DT ingests real telemetry from Kubernetes Metrics API, watch events for pod/node lifecycle, and chaos notifications via `/observe`. Telemetry collector runs continuously, updating node metrics and tracking pod completions.
2. **Simulate**: `PredictiveSimulator.score_plan()` evaluates candidate placements using calibrated compute/network models, architecture overhead factors, and origin-aware latency (from job origin cluster to candidate cluster).
3. **Act**: Selected plan is executed through Kubernetes API in the correct cluster; shadow placements are recorded for fast failover. Real workloads consume scaled resources (1:100 default).
4. **Verify**: Observed metrics collected from pod execution (latency, CPU, memory, energy) are compared against predictions. Verification results exported to CSV with error metrics and threshold compliance.

## 4. Simulation Methodology
- **Multi-cluster substrate**: `deploy/multi-cluster-setup.sh` creates 6-7 k3d clusters (dc-core, prosumer-mining, campus-lab, gamer-pc, phone-pan-1, phone-pan-2, edge-microdc), each with 2-3 nodes. KWOK config (`sim/kwok/cluster.yaml`) to scale logical nodes without heavy resource cost.
- **Inter-cluster latency**: `deploy/latency-matrix.yaml` defines latencies between clusters (e.g., dc-core <-> edge-microdc: 35ms). Policies use this for origin-aware placement.
- **Resource scaling**: Default 1:100 scaling (1 simulated CPU = 0.01 real cores) allows large-scale simulation on single machine. Configurable per experiment.
- **Heterogeneity**: NodeFeatureDiscovery labels (via `deploy/nfd-values.yaml`), multi-arch images, Wasm runtime class (`deploy/runtimeclass-wasm.yaml`).
- **Chaos & network**: Chaos Mesh workflows (`chaos/scenarios/*`), tc/netem daemonset, failure injection system (time-based and event-driven).
- **Calibration utilities**: `tools/calibrate_compute.py` (compute/energy proxy) and `tools/calibrate_network.py` (baseline RTT sampling).
- **Workloads**: `tools/load_generator.py`, `experiments/v1–v7`, `tools/trace_replay.py` for trace-driven validation. V7 dedicated to prediction verification.
- **Metrics**: JSON/CSV outputs under `reports/`, aggregated via `tools/analyze_results.py`. Verification results in `reports/verification/`.

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

