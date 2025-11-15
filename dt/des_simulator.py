from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple, TYPE_CHECKING

from dt.scaling import ResourceScaler, DEFAULT_SCALER

if TYPE_CHECKING:
    from dt.state import DTState, Job, JobStage, Node, PlacementDecision

# Baseline overhead factors used across the simulator and heuristic planners.
DEFAULT_QEMU_OVERHEAD = {
    "arm64": 0.30,
    "riscv64": 0.70,
    "amd64": 0.40,  # x86 emulated on non-x86 hosts
}


@dataclass(order=True)
class _Event:
    time_ms: float
    priority: int
    event_type: str = field(compare=False)
    stage_id: Optional[str] = field(compare=False, default=None)
    node_name: Optional[str] = field(compare=False, default=None)


@dataclass
class _StageExecution:
    stage: JobStage
    node_name: str
    exec_format: str
    start_ms: float
    finish_ms: Optional[float] = None
    energy_kwh: float = 0.0
    failed: bool = False


@dataclass
class DESMetrics:
    latency_ms: float
    energy_kwh: float
    risk_score: float
    completed_stages: int
    failed_stages: int
    sla_violated: bool


@dataclass
class _QueueEntry:
    stage: JobStage
    decision: PlacementDecision
    arrival_ms: float
    required_cores: int


class _NodeQueue:
    def __init__(self, node: Node) -> None:
        self.node = node
        self.total_cores = max(1, int(node.hardware.cpu_cores))
        self.available_cores = float(self.total_cores)
        self.waiting: Deque[_QueueEntry] = deque()

    def can_start(self, cores_needed: float) -> bool:
        return self.available_cores >= cores_needed

    def acquire(self, cores_needed: float) -> bool:
        if self.can_start(cores_needed):
            self.available_cores -= cores_needed
            return True
        return False

    def release(self, cores: float) -> None:
        self.available_cores = min(self.total_cores, self.available_cores + cores)


def compute_stage_runtime_ms(
    stage: JobStage,
    node: Node,
    exec_format: str,
    qemu_overhead: Optional[Dict[str, float]] = None,
) -> float:
    """Estimate runtime for a stage on a specific node."""
    overheads = qemu_overhead or DEFAULT_QEMU_OVERHEAD
    duration = max(1.0, float(stage.compute.duration_ms))
    frequency = max(0.1, float(node.hardware.base_ghz or 3.0))
    freq_scale = 3.0 / frequency
    runtime = duration * freq_scale

    if exec_format.startswith("qemu-"):
        target_arch = exec_format.split("-", 1)[1]
        runtime *= 1.0 + overheads.get(target_arch, 0.5)
    elif exec_format == "wasm":
        runtime *= 1.35

    workload = stage.compute.workload_type
    if workload == "io_bound":
        runtime *= 1.10
    elif workload == "gpu_bound" and node.hardware.gpu_vram_gb <= 0:
        # Penalise GPU workloads on CPU-only nodes.
        runtime *= 1.25

    # If the stage asks for more cores than the node has, stretch runtime.
    requested_cores = max(1, int(stage.compute.cpu))
    if requested_cores > node.hardware.cpu_cores:
        scale = requested_cores / max(1, node.hardware.cpu_cores)
        runtime *= scale

    return runtime


def compute_network_delay_ms(state: DTState, src_node: str, dst_node: str) -> float:
    if src_node == dst_node:
        return 0.0
    best = math.inf
    for link in state.links_incident_to(src_node):
        if link.a == dst_node or link.b == dst_node:
            best = min(best, link.base_latency_ms)
    return float(best if math.isfinite(best) else 10.0)


class DiscreteEventSimulator:
    """Event-driven simulation engine for staged jobs."""

    def __init__(
        self,
        state: DTState,
        *,
        qemu_overhead_map: Optional[Dict[str, float]] = None,
        failure_rate: float = 0.0,
        rng: Optional[random.Random] = None,
        scaler: Optional[ResourceScaler] = None,
    ) -> None:
        self.state = state
        self.qemu_overhead = qemu_overhead_map or DEFAULT_QEMU_OVERHEAD
        self.failure_rate = max(0.0, float(failure_rate))
        self.rng = rng or random.Random()
        self.scaler = scaler or DEFAULT_SCALER

        self._reset()

    def _reset(self) -> None:
        self.clock_ms = 0.0
        self._event_seq = 0
        self.events: list[_Event] = []
        self._placements: Dict[str, PlacementDecision] = {}
        self._nominal_duration = 0.0
        self.executions: Dict[str, _StageExecution] = {}
        self.node_queues: Dict[str, _NodeQueue] = {}
        self.remaining_deps: Dict[str, int] = {}
        self.dependents: Dict[str, list[str]] = {}
        self.total_energy_kwh = 0.0
        self.failed_stages: set[str] = set()
        self.completed_stages: set[str] = set()

    def simulate(self, job: Job, placements: Dict[str, PlacementDecision]) -> DESMetrics:
        self._reset()
        stage_map = {stage.id: stage for stage in job.stages}
        self._placements = placements
        self._nominal_duration = (
            sum(max(1.0, float(stage.compute.duration_ms)) for stage in job.stages)
            or 1000.0
        )

        # Guard against missing placements.
        for stage_id in stage_map.keys():
            if stage_id not in placements:
                raise ValueError(f"No placement provided for stage '{stage_id}'")

        self._build_dependency_graph(job)
        self._initialise_node_queues(placements)
        self._schedule_initial_ready_events(job)
        if self.failure_rate > 0.0:
            self._schedule_failures(placements)

        while self.events:
            event = heapq.heappop(self.events)
            self.clock_ms = event.time_ms
            if event.event_type == "stage_ready" and event.stage_id:
                self._handle_stage_ready(
                    stage_map[event.stage_id], placements[event.stage_id], event.time_ms
                )
            elif event.event_type == "stage_complete" and event.stage_id:
                self._handle_stage_complete(
                    stage_map[event.stage_id], placements[event.stage_id]
                )
            elif event.event_type == "node_failure" and event.node_name:
                self._handle_node_failure(event.node_name)

        return self._build_metrics(job)

    # ------------------------------------------------------------------ setup

    def _build_dependency_graph(self, job: Job) -> None:
        for stage in job.stages:
            self.remaining_deps.setdefault(stage.id, 0)
            if stage.predecessor:
                self.remaining_deps[stage.id] = 1
                self.dependents.setdefault(stage.predecessor, []).append(stage.id)

    def _initialise_node_queues(self, placements: Dict[str, PlacementDecision]) -> None:
        for decision in placements.values():
            if decision.node_name not in self.node_queues:
                node = self.state.get_node(decision.node_name)
                if node is None:
                    raise ValueError(f"Unknown node '{decision.node_name}' in placement")
                self.node_queues[decision.node_name] = _NodeQueue(node)

    def _schedule_initial_ready_events(self, job: Job) -> None:
        for stage in job.stages:
            if self.remaining_deps.get(stage.id, 0) == 0:
                self._push_event(0.0, "stage_ready", stage_id=stage.id)

    def _schedule_failures(self, placements: Dict[str, PlacementDecision]) -> None:
        if not placements:
            return
        nodes = list({dec.node_name for dec in placements.values()})
        for node_name in nodes:
            if self.rng.random() <= self.failure_rate:
                fraction = self.rng.uniform(0.1, 0.9)
                self._push_event(
                    fraction * self._nominal_duration, "node_failure", node_name=node_name
                )

    # ----------------------------------------------------------------- events

    def _handle_stage_ready(
        self, stage: JobStage, decision: PlacementDecision, ready_time: float
    ) -> None:
        queue = self.node_queues[decision.node_name]
        required_cores = max(1, int(stage.compute.cpu))
        entry = _QueueEntry(
            stage=stage,
            decision=decision,
            arrival_ms=ready_time,
            required_cores=required_cores,
        )

        if queue.acquire(required_cores):
            start_time = max(ready_time, self.clock_ms)
            self._start_stage(entry, queue, start_time)
        else:
            queue.waiting.append(entry)

    def _handle_stage_complete(self, stage: JobStage, decision: PlacementDecision) -> None:
        record = self.executions.get(stage.id)
        queue = self.node_queues[decision.node_name]
        required_cores = max(1, int(stage.compute.cpu))
        queue.release(required_cores)

        if record:
            record.finish_ms = self.clock_ms
            self.completed_stages.add(stage.id)
            self.total_energy_kwh += record.energy_kwh

        for dependent in self.dependents.get(stage.id, []):
            self.remaining_deps[dependent] = max(
                0, self.remaining_deps.get(dependent, 0) - 1
            )
            if self.remaining_deps[dependent] == 0:
                if record and record.failed:
                    self.failed_stages.add(dependent)
                    continue
                required_delay = self._dependency_delay(stage.id, dependent)
                self._push_event(
                    self.clock_ms + required_delay, "stage_ready", stage_id=dependent
                )

        self._drain_queue(queue)

    def _handle_node_failure(self, node_name: str) -> None:
        queue = self.node_queues.get(node_name)
        if not queue:
            return
        for exec_entry in list(self.executions.values()):
            if exec_entry.node_name == node_name and exec_entry.finish_ms is None:
                exec_entry.failed = True
                exec_entry.finish_ms = self.clock_ms
                self.failed_stages.add(exec_entry.stage.id)
        queue.waiting.clear()

    def _drain_queue(self, queue: _NodeQueue) -> None:
        while queue.waiting:
            entry = queue.waiting[0]
            if not queue.acquire(entry.required_cores):
                break
            queue.waiting.popleft()
            start_time = max(self.clock_ms, entry.arrival_ms)
            self._start_stage(entry, queue, start_time)

    def _start_stage(self, entry: _QueueEntry, queue: _NodeQueue, start_time: float) -> None:
        node = queue.node
        runtime_ms = compute_stage_runtime_ms(
            entry.stage, node, entry.decision.exec_format, self.qemu_overhead
        )
        duration_ms = max(1.0, runtime_ms)
        energy_kw = float(node.hardware.tdp_w or 95.0)
        energy_kwh = (energy_kw * (duration_ms / 1000.0)) / 3600.0

        record = _StageExecution(
            stage=entry.stage,
            node_name=node.name,
            exec_format=entry.decision.exec_format,
            start_ms=start_time,
            energy_kwh=energy_kwh,
        )
        self.executions[entry.stage.id] = record
        self._push_event(
            start_time + duration_ms,
            "stage_complete",
            stage_id=entry.stage.id,
            priority=1,
        )

    def _dependency_delay(self, predecessor_id: str, stage_id: str) -> float:
        prev_exec = self.executions.get(predecessor_id)
        next_decision = self._placements.get(stage_id)
        if not next_decision:
            return 0.0
        if not prev_exec or prev_exec.finish_ms is None:
            return 0.0
        return compute_network_delay_ms(
            self.state, prev_exec.node_name, next_decision.node_name
        )

    # ---------------------------------------------------------------- metrics

    def _build_metrics(self, job: Job) -> DESMetrics:
        all_stage_ids = {stage.id for stage in job.stages}
        incomplete = all_stage_ids - (self.completed_stages | self.failed_stages)
        self.failed_stages.update(incomplete)

        makespan = 0.0
        for execution in self.executions.values():
            if execution.finish_ms is not None:
                makespan = max(makespan, execution.finish_ms)

        latency_ms = makespan
        total_stages = max(1, len(all_stage_ids))
        risk_score = len(self.failed_stages) / total_stages
        sla_violated = bool(self.failed_stages) or (
            job.deadline_ms and latency_ms > job.deadline_ms
        )

        return DESMetrics(
            latency_ms=float(latency_ms),
            energy_kwh=float(self.total_energy_kwh),
            risk_score=float(risk_score),
            completed_stages=len(self.completed_stages),
            failed_stages=len(self.failed_stages),
            sla_violated=sla_violated,
        )

    # ---------------------------------------------------------------- helpers

    def _push_event(
        self,
        time_ms: float,
        event_type: str,
        *,
        stage_id: Optional[str] = None,
        node_name: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        self._event_seq += 1
        event = _Event(
            time_ms=time_ms,
            priority=priority,
            event_type=event_type,
            stage_id=stage_id,
            node_name=node_name,
        )
        heapq.heappush(self.events, event)

