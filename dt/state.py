from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import time
import threading
import ipaddress
from collections import defaultdict


@dataclass
class HardwareSpec:
        cpu_cores: int
        base_ghz: float
        memory_gb: int
        gpu_vram_gb: int = 0
        arch: str = "amd64"  # amd64 | arm64 | riscv64
        uarch: Optional[str] = None
        tdp_w: Optional[float] = None
        memory_mhz: Optional[int] = None
        memory_type: Optional[str] = None
        accelerators: List[str] = field(default_factory=list)


@dataclass
class NodeRuntime:
        native_formats: List[str] = field(default_factory=lambda: ["native"])
        emulation_support: List[str] = field(default_factory=list)  # e.g., ["arm64","riscv64"]
        wasm_support: bool = False


@dataclass
class KubernetesInfo:
        node_name: str
        labels: Dict[str, str] = field(default_factory=dict)
        allocatable_cpu: float = 0.0
        allocatable_mem_gb: float = 0.0
        zone: Optional[str] = None


@dataclass
class Telemetry:
        cpu_util: float = 0.0
        mem_util: float = 0.0
        net_rx_mbps: float = 0.0
        net_tx_mbps: float = 0.0
        cpu_temp_c: Optional[float] = None
        last_heartbeat_s: float = field(default_factory=time.time)


@dataclass
class Node:
        name: str
        hardware: HardwareSpec
        runtime: NodeRuntime
        k8s: KubernetesInfo
        tel: Telemetry = field(default_factory=Telemetry)
        available: bool = True
        network: Optional["NetworkIdentity"] = None
        pool_key: Optional[str] = None
        pool_score: float = 0.0


@dataclass
class Link:
        a: str
        b: str
        base_latency_ms: float
        bandwidth_gbps: float
        loss_pct: float = 0.0


@dataclass
class StageCompute:
        cpu: int
        mem_gb: int
        duration_ms: int
        gpu_vram_gb: int = 0
        workload_type: str = "cpu_bound"  # cpu_bound | io_bound | gpu_bound


@dataclass
class StageConstraints:
        arch: List[str] = field(default_factory=lambda: ["amd64"])
        formats: List[str] = field(default_factory=lambda: ["native"])
        data_locality: Optional[str] = None
        max_latency_to_predecessor_ms: Optional[int] = None


@dataclass
class JobStage:
        id: str
        compute: StageCompute
        constraints: StageConstraints
        predecessor: Optional[str] = None


@dataclass
class JobOrigin:
        """Origin context for a job request."""
        cluster: str
        node: Optional[str] = None


@dataclass
class Job:
        name: str
        deadline_ms: int
        stages: List[JobStage]
        origin: Optional[JobOrigin] = None


@dataclass
class PlacementDecision:
        stage_id: str
        node_name: str
        exec_format: str  # native | qemu-<arch> | wasm | cuda


@dataclass
class ObservedMetrics:
        """Observed metrics for a plan execution."""
        latency_ms: float
        cpu_util: float = 0.0
        mem_peak_gb: float = 0.0
        energy_kwh: float = 0.0
        completed_at: Optional[float] = None  # timestamp


@dataclass
class ClusterInfo:
        """Information about a cluster."""
        name: str
        cluster_type: str  # datacenter, mining, lab, gaming, pan, edge
        resiliency_score: float = 0.8
        nodes: List[str] = field(default_factory=list)
        cluster_id: Optional[int] = None
        network_cidr: Optional[str] = None
        pod_cidr: Optional[str] = None
        service_cidr: Optional[str] = None
        node_pools: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class Plan:
        plan_id: str
        job_name: str
        placements: Dict[str, PlacementDecision]
        predicted_latency_ms: float
        predicted_energy_kwh: float
        risk_score: float
        shadow_plan: Dict[str, str] = field(default_factory=dict)  # stage_backup -> node


@dataclass
class NetworkIdentity:
        """Virtual network coordinates for a node."""

        cluster_id: int
        cluster_name: str
        node_id: int
        network_cidr: str
        pod_cidr: str
        service_cidr: str
        node_ip: str
        pod_ip: str
        mac_address: str


@dataclass
class NodePool:
        """Capability-aware grouping of nodes inside a cluster."""

        cluster_id: int
        cluster_name: str
        pool_key: str
        characteristics: Dict[str, Any]
        nodes: List[str] = field(default_factory=list)
        score: float = 0.0


class VirtualTopology:
        """Allocator for virtual cluster/node identifiers and pools."""

        def __init__(self) -> None:
                self._cluster_id_seq: int = 1
                self._cluster_ids: Dict[str, int] = {}
                self._cluster_networks: Dict[str, Dict[str, str]] = {}
                self._node_counters: Dict[int, int] = defaultdict(int)
                self._identities: Dict[str, NetworkIdentity] = {}
                self._pools: Dict[Tuple[int, str], NodePool] = {}

        # ------------------------------------------------------------------
        # Cluster registration & addressing helpers
        # ------------------------------------------------------------------
        def register_cluster(self, cluster: ClusterInfo) -> int:
                """Ensure a cluster has a stable 8-bit identifier and subnets."""

                if cluster.name in self._cluster_ids:
                        cluster_id = self._cluster_ids[cluster.name]
                else:
                        if self._cluster_id_seq > 255:
                                raise ValueError("exhausted virtual cluster id space (8-bit)")
                        cluster_id = self._cluster_id_seq
                        self._cluster_id_seq += 1
                        self._cluster_ids[cluster.name] = cluster_id

                network = self._cluster_networks.get(cluster.name)
                if not network:
                        network = self._build_network_profile(cluster, cluster_id)
                        self._cluster_networks[cluster.name] = network

                cluster.cluster_id = cluster_id
                cluster.network_cidr = cluster.network_cidr or network["network_cidr"]
                cluster.pod_cidr = cluster.pod_cidr or network["pod_cidr"]
                cluster.service_cidr = cluster.service_cidr or network["service_cidr"]
                cluster.node_pools = cluster.node_pools or {}

                return cluster_id

        def ensure_node(self, cluster_name: str, node: Node) -> Tuple[Optional[NetworkIdentity], Optional[NodePool]]:
                """Allocate a virtual node identity and pool membership."""

                cluster_id = self._cluster_ids.get(cluster_name)
                network = self._cluster_networks.get(cluster_name)
                if cluster_id is None or not network:
                        return None, None

                # Reuse existing assignment when possible
                existing = self._identities.get(node.name)
                if existing and existing.cluster_id == cluster_id:
                        pool = None
                        for (cid, key), candidate in self._pools.items():
                                if cid == cluster_id and node.name in candidate.nodes:
                                        pool = candidate
                                        node.pool_key = key
                                        node.pool_score = candidate.score
                                        break
                        return existing, pool

                node_id = self._allocate_node_id(cluster_id, node)
                pod_ip = self._allocate_ip(network["pod_cidr"], node_id)
                node_ip = self._allocate_ip(network["network_cidr"], node_id)
                mac = self._derive_mac(cluster_id, node_id)

                identity = NetworkIdentity(
                        cluster_id=cluster_id,
                        cluster_name=cluster_name,
                        node_id=node_id,
                        network_cidr=network["network_cidr"],
                        pod_cidr=network["pod_cidr"],
                        service_cidr=network["service_cidr"],
                        node_ip=node_ip,
                        pod_ip=pod_ip,
                        mac_address=mac,
                )
                self._identities[node.name] = identity

                pool_key, characteristics, score = self._classify_node(node)
                pool = self._pools.get((cluster_id, pool_key))
                if not pool:
                        pool = NodePool(
                                cluster_id=cluster_id,
                                cluster_name=cluster_name,
                                pool_key=pool_key,
                                characteristics=characteristics,
                                nodes=[],
                                score=score,
                        )
                        self._pools[(cluster_id, pool_key)] = pool

                if node.name not in pool.nodes:
                        pool.nodes.append(node.name)
                # Keep aggregate score meaningful (max captures pool capability envelope)
                pool.score = max(pool.score, score)
                pool.characteristics.update(characteristics)

                node.pool_key = pool_key
                node.pool_score = score

                return identity, pool

        def get_identity(self, node_name: str) -> Optional[NetworkIdentity]:
                return self._identities.get(node_name)

        def list_pools(self) -> List[NodePool]:
                return [NodePool(
                        cluster_id=p.cluster_id,
                        cluster_name=p.cluster_name,
                        pool_key=p.pool_key,
                        characteristics=dict(p.characteristics),
                        nodes=list(p.nodes),
                        score=p.score,
                ) for p in self._pools.values()]

        def describe(self) -> Dict[str, Any]:
                clusters: Dict[str, Any] = {}
                for name, cid in self._cluster_ids.items():
                        network = self._cluster_networks.get(name, {})
                        clusters[name] = {
                                "cluster_id": cid,
                                "network_cidr": network.get("network_cidr"),
                                "pod_cidr": network.get("pod_cidr"),
                                "service_cidr": network.get("service_cidr"),
                                "nodes": [],
                                "pools": {},
                        }

                for node_name, identity in self._identities.items():
                        cluster = clusters.setdefault(identity.cluster_name, {
                                "cluster_id": identity.cluster_id,
                                "network_cidr": identity.network_cidr,
                                "pod_cidr": identity.pod_cidr,
                                "service_cidr": identity.service_cidr,
                                "nodes": [],
                                "pools": {},
                        })
                        cluster["nodes"].append({
                                "name": node_name,
                                "node_id": identity.node_id,
                                "pod_ip": identity.pod_ip,
                                "node_ip": identity.node_ip,
                                "mac_address": identity.mac_address,
                        })

                for pool in self._pools.values():
                        cluster = clusters.setdefault(pool.cluster_name, {
                                "cluster_id": pool.cluster_id,
                                "network_cidr": None,
                                "pod_cidr": None,
                                "service_cidr": None,
                                "nodes": [],
                                "pools": {},
                        })
                        cluster["pools"][pool.pool_key] = {
                                "score": pool.score,
                                "characteristics": dict(pool.characteristics),
                                "nodes": list(pool.nodes),
                        }

                return clusters

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------
        def _build_network_profile(self, cluster: ClusterInfo, cluster_id: int) -> Dict[str, str]:
                pod_cidr = cluster.pod_cidr or f"10.{cluster_id}.0.0/16"
                service_cidr = cluster.service_cidr or f"10.{cluster_id}.128.0/17"
                network_cidr = cluster.network_cidr or f"172.30.{cluster_id}.0/24"
                return {
                        "pod_cidr": pod_cidr,
                        "service_cidr": service_cidr,
                        "network_cidr": network_cidr,
                }

        def _allocate_node_id(self, cluster_id: int, node: Node) -> int:
                label = None
                if node.k8s and node.k8s.labels:
                        label = node.k8s.labels.get("dt.virtual.node_id")
                if label:
                        try:
                                node_id = int(label)
                                self._node_counters[cluster_id] = max(self._node_counters[cluster_id], node_id)
                                return node_id
                        except ValueError:
                                pass

                self._node_counters[cluster_id] += 1
                node_id = self._node_counters[cluster_id]
                if node_id > 0xFFFF:
                        raise ValueError("exhausted virtual node id space (16-bit)")
                return node_id

        def _allocate_ip(self, cidr: str, node_id: int) -> str:
                network = ipaddress.ip_network(cidr, strict=False)
                host_index = max(10, node_id + 10)  # Reserve early addresses for infra
                if host_index >= network.num_addresses - 1:
                        host_index = (node_id % (network.num_addresses - 2)) + 1
                ip_addr = network.network_address + host_index
                if ip_addr == network.broadcast_address:
                        ip_addr -= 1
                return str(ip_addr)

        def _derive_mac(self, cluster_id: int, node_id: int) -> str:
                return "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}".format(
                        0x02,
                        0xFD,
                        cluster_id & 0xFF,
                        (node_id >> 8) & 0xFF,
                        node_id & 0xFF,
                        (node_id * 37) & 0xFF,
                )

        def _classify_node(self, node: Node) -> Tuple[str, Dict[str, Any], float]:
                hw = node.hardware
                cores = hw.cpu_cores or 0
                base_ghz = hw.base_ghz or 0.0
                tdp = hw.tdp_w or (cores * base_ghz * 12)
                core_score = cores * max(base_ghz, 1.0)

                if cores >= 48 or tdp >= 220:
                        core_tier = "compute-ultra"
                elif cores >= 24 or tdp >= 140:
                        core_tier = "compute-high"
                elif cores >= 12 or tdp >= 95:
                        core_tier = "compute-balanced"
                else:
                        core_tier = "compute-edge"

                mem_speed = hw.memory_mhz or 2666
                mem_total = hw.memory_gb or 0
                if mem_speed >= 5600 or (hw.memory_type or "").lower().startswith("ddr5"):
                        mem_tier = "mem-ddr5-elite"
                elif mem_speed >= 4266:
                        mem_tier = "mem-ddr4-ultra"
                elif mem_speed >= 3600:
                        mem_tier = "mem-ddr4-fast"
                elif mem_speed >= 2666:
                        mem_tier = "mem-mainstream"
                else:
                        mem_tier = "mem-legacy"

                if hw.accelerators:
                        kinds = sorted({acc.lower() for acc in hw.accelerators})
                        accel_tier = "accel-" + "-".join(kinds)
                elif hw.gpu_vram_gb:
                        accel_tier = "accel-gpu"
                else:
                        accel_tier = "accel-none"

                pool_key = f"{core_tier}|{mem_tier}|{accel_tier}"
                characteristics = {
                        "core_tier": core_tier,
                        "mem_tier": mem_tier,
                        "accelerator_tier": accel_tier,
                        "cpu_cores": cores,
                        "base_ghz": base_ghz,
                        "tdp_w": tdp,
                        "memory_gb": mem_total,
                        "memory_mhz": mem_speed,
                        "accelerators": list(hw.accelerators),
                        "gpu_vram_gb": hw.gpu_vram_gb,
                }

                score = (core_score * 1.5) + (mem_total * 0.75) + (mem_speed / 100.0)
                if accel_tier != "accel-none":
                        score += 150.0

                return pool_key, characteristics, score



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt/state.py — Digital Twin runtime state for the Fabric simulator.

Responsibilities
---------------
- Load per-node descriptors:          ./nodes/*.yaml
- Load optional topology:             ./sim/topology.yaml  (links, defaults)
- Watch & merge runtime overrides:    ./sim/overrides.json (written by sim/chaos.py)
- Maintain thread-safe resource view: capacities, reservations, queues
- Offer a compact API for dt/api.py:
    • snapshot()                 → dict (nodes, links, ts)
    • reserve(req)               → reservation_id or None
    • release(reservation_id)    → bool
    • score_node_basic(stage,n)  → float (lower is better)
    • apply_observation(payload) → merge ad-hoc updates (used by /observe)

Design notes
------------
- No hard dependency on Flask here (pure state). dt/api.py can import and call it.
- Only non-stdlib dep is PyYAML. (requests is optional if you later push updates out.)
- Links are stored as an undirected map keyed by "A|B".
- Dynamic/ephemeral state is kept under node["dyn"] and link["dyn"].

Paths (configurable via constructor)
-----------------------------------
nodes_dir        default: "nodes"
topology_path    default: "sim/topology.yaml" (optional)
overrides_path   default: "sim/overrides.json" (optional)

"""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# from .events import EventBus, build_cloudevent  # Module not implemented yet
from .predict import NodeForecast, PredictiveAnalyzer


# ----------------------------- helpers -----------------------------

def link_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def utc_ms() -> int:
    return int(time.time() * 1000)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------- data classes -----------------------------

@dataclass
class NodeDyn:
    """Mutable, runtime-only fields for a node."""
    down: bool = False
    thermal_derate: float = 0.0         # 0..1
    power_cap_w: Optional[float] = None
    clock_skew_ms: Optional[float] = None
    packet_dup: Optional[float] = None
    packet_reorder: Optional[float] = None
    used_cpu_cores: float = 0.0
    used_mem_gb: float = 0.0
    used_gpu_vram_gb: float = 0.0
    reliability: float = 0.95
    availability_window_sec: Optional[float] = None
    mtbf_hours: Optional[float] = None
    uptime_hours: Optional[float] = None
    battery_pct: Optional[float] = None
    battery_drain_pct_per_hr: Optional[float] = None
    util_forecast: float = 0.0
    projected_derate: float = 0.0
    predicted_failure_window_sec: Optional[float] = None
    reservations: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # res_id -> req


@dataclass
class LinkDyn:
    """Mutable, runtime-only fields for a link."""
    down: bool = False
    speed_gbps: Optional[float] = None
    rtt_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    loss_pct: Optional[float] = None
    ecn: Optional[bool] = None
    latency_p95_ms: Optional[float] = None


# ----------------------------- DT State -----------------------------

class DTState:
    def __init__(
        self,
        nodes_dir: str = "nodes",
        topology_path: str = "sim/topology.yaml",
        overrides_path: str = "sim/overrides.json",
        watch_interval_sec: float = 0.5,
        auto_start_watchers: bool = True,
    ):
        self.nodes_dir = Path(nodes_dir)
        self.topology_path = Path(topology_path)
        self.overrides_path = Path(overrides_path)

        self._lock = threading.RLock()
        # Lightweight compatibility attributes expected by the simplified
        # planner/API integration.
        self.clusters: Dict[str, ClusterInfo] = {}
        self.observed_metrics: Dict[str, ObservedMetrics] = {}
        self._jobs: Dict[str, Job] = {}
        self._virtual = VirtualTopology()
        event_buf = max(32, safe_int(os.environ.get("FABRIC_DT_EVENT_BUFFER", 512), 512))
        # self._events = EventBus(maxlen=event_buf)  # EventBus not implemented yet
        # Simple stub for events
        class EventStub:
            def __init__(self, maxlen):
                self.events = []
                self.maxlen = maxlen
            def emit(self, evt):
                self.events.append(evt)
                if len(self.events) > self.maxlen:
                    self.events.pop(0)
            def recent(self, limit=100, since_id=None):
                return self.events[-limit:] if since_id is None else [e for e in self.events if e.get("id") != since_id][-limit:]
        self._events = EventStub(event_buf)
        self._predictor = PredictiveAnalyzer()
        self._snapshot_cache: Optional[Dict[str, Any]] = None
        self._snapshot_generation: int = 0

        # Static-ish structures
        self.nodes_by_name: Dict[str, Dict[str, Any]] = {}  # includes 'dyn'
        self.links_by_key: Dict[str, Dict[str, Any]] = {}   # includes 'dyn'
        self.defaults: Dict[str, Any] = {}
        self._nodes_fingerprint: Dict[str, float] = {}

        # Overrides (raw copies of sim/overrides.json)
        self._overrides: Dict[str, Any] = {"nodes": {}, "links": {}}
        self._overrides_mtime: float = 0.0

        # Node/Topology mtimes to allow hot reloads if you want to extend it
        self._nodes_mtime: float = 0.0
        self._topology_mtime: float = 0.0

        # Reservation counter
        self._res_seq: int = 1

        # Initial load
        self._load_nodes_locked()
        self._load_topology_locked()
        self._load_overrides_locked(apply_now=True)
        self._refresh_virtual_topology_from_nodes_locked()

        # Background watcher for overrides (and optionally hot-reload topology)
        self._watch_interval = max(0.2, float(watch_interval_sec))
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if auto_start_watchers:
            self.start()

    # -------- public lifecycle --------

    def start(self):
        if self._watch_thread and self._watch_thread.is_alive():
            return
        self._stop_event.clear()
        self._watch_thread = threading.Thread(target=self._watch_loop, name="DTStateWatch", daemon=True)
        self._watch_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=2.0)

    # -------- loads & merges --------

    def _load_nodes_locked(self, preserve_dyn: bool = True) -> bool:
        """Load ./nodes/*.yaml into nodes_by_name with fresh dyn slots."""
        with self._lock:
            nodes: Dict[str, Dict[str, Any]] = {}
            latest_mtime = self._nodes_mtime
            file_stats = []
            try:
                for f in sorted(self.nodes_dir.glob("*.yaml")):
                    try:
                        stat = f.stat()
                    except FileNotFoundError:
                        # File disappeared between glob and stat; ignore this round
                        continue
                    file_stats.append((f, stat))
            except FileNotFoundError:
                file_stats = []

            new_fingerprint = {f.name: stat.st_mtime for f, stat in file_stats}
            if preserve_dyn and new_fingerprint == self._nodes_fingerprint:
                return False

            for f, stat in file_stats:
                try:
                    latest_mtime = max(latest_mtime, stat.st_mtime)
                    data = yaml.safe_load(f.read_text(encoding="utf-8"))
                    name = data.get("name")
                    if not name:
                        continue
                    # Ensure dyn exists and keep capacity-derived caches
                    data.setdefault("dyn", NodeDyn().__dict__.copy())
                    # Cached capacities
                    self._compute_and_cache_capacities(data)
                    health = data.get("health") or {}
                    lifecycle = data.get("lifecycle") or {}
                    power = data.get("power") or {}
                    dyn = data.get("dyn") or {}
                    reliability = health.get("reliability")
                    availability = lifecycle.get("availability_window_sec")
                    mtbf = health.get("mtbf_hours")
                    uptime = health.get("uptime_hours")
                    battery_pct = power.get("battery_pct")
                    battery_drain = power.get("battery_drain_pct_per_hr")
                    dyn.setdefault("reliability", safe_float(reliability, 0.95))
                    dyn.setdefault("availability_window_sec", safe_float(availability, None))
                    dyn.setdefault("mtbf_hours", safe_float(mtbf, None))
                    dyn.setdefault("uptime_hours", safe_float(uptime, None))
                    dyn.setdefault("battery_pct", None if battery_pct is None else safe_float(battery_pct, None))
                    dyn.setdefault(
                        "battery_drain_pct_per_hr",
                        None if battery_drain is None else safe_float(battery_drain, None),
                    )
                    prev = self.nodes_by_name.get(name) if preserve_dyn else None

                    dyn_defaults = NodeDyn().__dict__.copy()
                    disk_dyn = data.get("dyn") or {}
                    for key, value in disk_dyn.items():
                        if key in dyn_defaults:
                            if key == "reservations" and isinstance(value, dict):
                                dyn_defaults[key] = dict(value)
                            else:
                                dyn_defaults[key] = value

                    if prev:
                        prev_dyn = prev.get("dyn", {}) or {}
                        for key, value in prev_dyn.items():
                            if key == "reservations" and isinstance(value, dict):
                                dyn_defaults[key] = dict(value)
                            elif key in dyn_defaults:
                                dyn_defaults[key] = value

                    data["dyn"] = dyn_defaults

                    self._predictor.ensure_node(
                        name,
                        reliability=None if reliability is None else safe_float(reliability, 0.95),
                        availability_window_sec=None if availability is None else safe_float(availability, 0.0),
                        battery_pct=None if battery_pct is None else safe_float(battery_pct, 0.0),
                        battery_drain_pct_per_hr=None
                        if battery_drain is None
                        else safe_float(battery_drain, 0.0),
                        mtbf_hours=None if mtbf is None else safe_float(mtbf, 0.0),
                        uptime_hours=None if uptime is None else safe_float(uptime, 0.0),
                    )
                    nodes[name] = data
                except Exception as e:
                    print(f"[state] WARN: failed to load node {f.name}: {e}")

            self.nodes_by_name = nodes
            self._nodes_mtime = latest_mtime
            self._nodes_fingerprint = new_fingerprint
            for node_name in self.nodes_by_name.keys():
                self._update_predictive_for_node_locked(node_name)
            self._invalidate_snapshot_locked()
            return True

    def _load_topology_locked(self):
        """Load topology (links + defaults) if present."""
        with self._lock:
            if not self.topology_path.exists():
                self.links_by_key = {}
                self.defaults = {}
                self._invalidate_snapshot_locked()
                return
            try:
                stat = self.topology_path.stat()
                topo = yaml.safe_load(self.topology_path.read_text(encoding="utf-8"))
                self._topology_mtime = stat.st_mtime

                # Defaults (optional; used if you want to fall back)
                self.defaults = topo.get("defaults", {}) or {}

                links: Dict[str, Dict[str, Any]] = {}
                for ln in (topo.get("links") or []):
                    a, b = ln.get("a"), ln.get("b")
                    if not a or not b:
                        continue
                    k = link_key(a, b)
                    lnd = {
                        "a": a, "b": b,
                        "profile": ln.get("profile"),
                        "qos_class": ln.get("qos_class"),
                        "scope": ln.get("scope", "site"),
                        "subnet": ln.get("subnet"),
                        "base": {
                            # Allow explicit metrics in link inline
                            "speed_gbps": ln.get("speed_gbps"),
                            "rtt_ms": ln.get("rtt_ms"),
                            "jitter_ms": ln.get("jitter_ms"),
                            "loss_pct": ln.get("loss_pct"),
                            "ecn": ln.get("ecn"),
                        },
                        "dyn": LinkDyn().__dict__.copy(),
                    }
                    # Strip Nones from base for cleanliness
                    lnd["base"] = {k2: v2 for k2, v2 in lnd["base"].items() if v2 is not None}
                    links[k] = lnd
                    self._predictor.ensure_link(k)
                self.links_by_key = links
                for key in list(self.links_by_key.keys()):
                    self._update_link_predictive_locked(key)
                self._invalidate_snapshot_locked()
            except Exception as e:
                print(f"[state] WARN: failed to load topology: {e}")
                self.links_by_key = {}
                self.defaults = {}
                self._invalidate_snapshot_locked()

    def _load_overrides_locked(self, apply_now: bool = True):
        """Load sim/overrides.json if present; optionally apply immediately."""
        with self._lock:
            if not self.overrides_path.exists():
                self._overrides = {"nodes": {}, "links": {}}
                self._overrides_mtime = 0.0
                return
            try:
                stat = self.overrides_path.stat()
                if stat.st_mtime <= self._overrides_mtime:
                    return
                raw = json.loads(self.overrides_path.read_text(encoding="utf-8"))
                self._overrides = {
                    "nodes": raw.get("nodes", {}) or {},
                    "links": raw.get("links", {}) or {},
                }
                self._overrides_mtime = stat.st_mtime
                if apply_now:
                    self._apply_overrides_locked()
            except Exception as e:
                print(f"[state] WARN: failed to load overrides.json: {e}")

    def _apply_overrides_locked(self):
        """Merge self._overrides into node/link dyn fields."""
        # Nodes
        for nname, changes in self._overrides.get("nodes", {}).items():
            n = self.nodes_by_name.get(nname)
            if not n:
                continue
            dyn = n.setdefault("dyn", NodeDyn().__dict__.copy())
            # Only accept known fields
            for k in ("down", "power_cap_w", "thermal_derate", "clock_skew_ms",
                      "packet_dup", "packet_reorder"):
                if k in changes:
                    dyn[k] = changes[k]
            if any(key in changes for key in ("down", "thermal_derate")):
                self._update_predictive_for_node_locked(nname)

        # Links
        for k, changes in self._overrides.get("links", {}).items():
            l = self.links_by_key.get(k)
            if not l:
                # Permit ad-hoc links (e.g., node↔node Wi-Fi), create shell
                parts = k.split("|", 1)
                if len(parts) == 2:
                    l = {"a": parts[0], "b": parts[1], "base": {}, "dyn": LinkDyn().__dict__.copy()}
                    self.links_by_key[k] = l
                else:
                    continue
            dyn = l.setdefault("dyn", LinkDyn().__dict__.copy())
            for kk in ("down", "speed_gbps", "rtt_ms", "jitter_ms", "loss_pct", "ecn"):
                if kk in changes:
                    dyn[kk] = changes[kk]
            if any(key in changes for key in ("speed_gbps", "rtt_ms", "jitter_ms", "loss_pct", "down")):
                self._update_link_predictive_locked(k)

        self._invalidate_snapshot_locked()

    def _emit_event(self, event_type: str, data: Dict[str, Any], subject: Optional[str] = None) -> None:
        # evt = build_cloudevent(event_type, "fabric.dt.state", data, subject=subject)  # Not implemented yet
        evt = {"type": event_type, "source": "fabric.dt.state", "data": data, "subject": subject}
        self._events.emit(evt)

    def _update_predictive_for_node_locked(self, name: str) -> Optional[NodeForecast]:
        node = self.nodes_by_name.get(name)
        if not node:
            return None
        dyn = node.setdefault("dyn", NodeDyn().__dict__.copy())
        caps = node.get("caps", {})
        max_cores = safe_float(caps.get("max_cpu_cores"), 0.0)
        used = safe_float(dyn.get("used_cpu_cores"), 0.0)
        util = 0.0 if max_cores <= 1e-9 else clamp(used / max(1e-9, max_cores), 0.0, 1.0)
        forecast = self._predictor.record_node_util(
            name,
            util,
            thermal_derate=safe_float(dyn.get("thermal_derate"), 0.0),
            reliability=dyn.get("reliability"),
            availability_window_sec=dyn.get("availability_window_sec"),
            battery_pct=dyn.get("battery_pct"),
            battery_drain_pct_per_hr=dyn.get("battery_drain_pct_per_hr"),
            mtbf_hours=dyn.get("mtbf_hours"),
            uptime_hours=dyn.get("uptime_hours"),
        )
        dyn["util_forecast"] = forecast.util_forecast
        dyn["projected_derate"] = forecast.projected_derate
        dyn["reliability"] = forecast.reliability
        dyn["predicted_failure_window_sec"] = forecast.availability_window_sec
        event_payload = {
            "node": name,
            "util_now": forecast.util_now,
            "util_forecast": forecast.util_forecast,
            "reliability": forecast.reliability,
            "availability_window_sec": forecast.availability_window_sec,
            "projected_derate": forecast.projected_derate,
        }
        self._emit_event("fabric.node.update", event_payload, subject=name)
        return forecast

    def _update_link_predictive_locked(self, key: str) -> None:
        link = self.links_by_key.get(key)
        if not link:
            return
        eff = self._effective_link(link)
        forecast = self._predictor.record_link_metrics(
            key,
            latency_ms=safe_float(eff.get("rtt_ms"), 0.0),
            jitter_ms=safe_float(eff.get("jitter_ms"), 0.0),
            loss_pct=safe_float(eff.get("loss_pct"), 0.0),
        )
        dyn = link.setdefault("dyn", LinkDyn().__dict__.copy())
        dyn["latency_p95_ms"] = forecast.latency_p95_ms
        dyn.setdefault("rtt_ms", forecast.latency_ms)
        dyn.setdefault("jitter_ms", forecast.jitter_ms)
        dyn.setdefault("loss_pct", forecast.loss_pct)
        event_payload = {
            "link": key,
            "latency_ms": forecast.latency_ms,
            "jitter_ms": forecast.jitter_ms,
            "loss_pct": forecast.loss_pct,
            "latency_p95_ms": forecast.latency_p95_ms,
        }
        self._emit_event("fabric.link.update", event_payload, subject=key)

    def _compute_and_cache_capacities(self, node: Dict[str, Any]):
        """Precompute static capacities and store under node['caps']."""
        cpu = node.get("cpu", {}) or {}
        mem = node.get("memory", {}) or {}
        gpu = node.get("gpu", {}) or {}

        cores = safe_float(cpu.get("cores"), 0.0)
        base_ghz = safe_float(cpu.get("base_ghz"), 0.0)
        ram_gb = safe_float(mem.get("ram_gb"), 0.0)
        vram_gb = safe_float(gpu.get("vram_gb"), 0.0)

        # naive "capacity units"
        cpu_units = cores * base_ghz
        node["caps"] = {
            "cpu_units": cpu_units,
            "max_cpu_cores": cores,
            "ram_gb": ram_gb,
            "gpu_vram_gb": vram_gb,
        }

    # -------- watcher loop --------

    def _watch_loop(self):
        while not self._stop_event.is_set():
            try:
                # Overrides
                self._load_overrides_locked(apply_now=True)

                # (Optional) Hot-reload topology if changed on disk
                if self.topology_path.exists():
                    stat = self.topology_path.stat()
                    if stat.st_mtime > self._topology_mtime:
                        self._load_topology_locked()

                # Hot-reload nodes when descriptors change on disk
                nodes_changed = self._load_nodes_locked(preserve_dyn=True)
                if nodes_changed:
                    with self._lock:
                        self._apply_overrides_locked()
            except Exception as e:
                print(f"[state] WARN: watcher iteration failed: {e}")

            self._stop_event.wait(self._watch_interval)

    # -------- public API (read) --------

    def _build_snapshot_locked(self) -> Dict[str, Any]:
        overview = self._predictor.overview()
        nodes = []
        predictive_nodes = overview.get("nodes", {})
        for n in self.nodes_by_name.values():
            dyn = n.get("dyn", {})
            caps = n.get("caps", {})
            eff = self._effective_caps(n)
            forecast = predictive_nodes.get(n.get("name"), {})
            merged_dyn = dict(dyn)
            if forecast:
                merged_dyn.setdefault("util_forecast", forecast.get("util_forecast"))
                merged_dyn.setdefault("projected_derate", forecast.get("projected_derate"))
                merged_dyn.setdefault("reliability", forecast.get("reliability"))
                merged_dyn.setdefault("predicted_failure_window_sec", forecast.get("availability_window_sec"))
            nodes.append({
                "name": n.get("name"),
                "class": n.get("class"),
                "arch": n.get("arch"),
                "formats_supported": n.get("formats_supported", []),
                "labels": n.get("labels", {}),
                "network": n.get("network", {}),
                "gpu": n.get("gpu", {}),
                "caps": caps,
                "dyn": merged_dyn,
                "effective": eff,
            })

        links = []
        predictive_links = overview.get("links", {})
        for k, l in self.links_by_key.items():
            eff_link = self._effective_link(l)
            forecast = predictive_links.get(k, {})
            dyn = dict(l.get("dyn", {}))
            if forecast:
                dyn.setdefault("latency_p95_ms", forecast.get("latency_p95_ms"))
                if forecast.get("latency_ms") is not None:
                    dyn.setdefault("rtt_ms", forecast.get("latency_ms"))
                if forecast.get("jitter_ms") is not None:
                    dyn.setdefault("jitter_ms", forecast.get("jitter_ms"))
                if forecast.get("loss_pct") is not None:
                    dyn.setdefault("loss_pct", forecast.get("loss_pct"))
            links.append({
                "key": k,
                "a": l.get("a"),
                "b": l.get("b"),
                "base": l.get("base", {}),
                "dyn": dyn,
                "effective": eff_link,
            })

        federations, federation_links, node_federations = self._federation_overview_locked()

        snapshot = {
            "ts": utc_ms(),
            "nodes": nodes,
            "links": links,
            "federations": federations,
            "federation_links": federation_links,
            "node_federations": node_federations,
            "predictive": overview,
        }
        return snapshot

    def _invalidate_snapshot_locked(self) -> None:
        self._snapshot_cache = None

    def snapshot(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot for UI/clients."""
        with self._lock:
            if self._snapshot_cache is None:
                self._snapshot_cache = self._build_snapshot_locked()
            return copy.deepcopy(self._snapshot_cache)

    def get_node(self, name: str) -> Optional[Node]:
        """Get a Node object by name, converting from internal dict format."""
        with self._lock:
            node_dict = self.nodes_by_name.get(name)
            if not node_dict:
                return None
            return self._node_from_dict_locked(name, node_dict)

    def _node_from_dict_locked(self, name: str, node_dict: Dict[str, Any]) -> Optional[Node]:
        try:
            hardware = node_dict.get("hardware") or {}
            runtime = node_dict.get("runtime") or {}
            k8s = node_dict.get("k8s") or {}
            pool_info = node_dict.get("pool") or {}
            network_info = node_dict.get("network") or {}

            hw = HardwareSpec(
                cpu_cores=int(hardware.get("cpu_cores", 0)),
                base_ghz=float(hardware.get("base_ghz", 0.0)),
                memory_gb=int(hardware.get("memory_gb", 0)),
                gpu_vram_gb=int(hardware.get("gpu_vram_gb", 0)),
                arch=hardware.get("arch", "amd64"),
                tdp_w=hardware.get("tdp_w"),
                memory_mhz=hardware.get("memory_mhz"),
                memory_type=hardware.get("memory_type"),
                accelerators=list(hardware.get("accelerators", [])),
            )
            rt = NodeRuntime(
                native_formats=list(runtime.get("native_formats", ["native"])),
                emulation_support=list(runtime.get("emulation_support", [])),
                wasm_support=bool(runtime.get("wasm_support", False)),
            )
            k8s_info = None
            if k8s:
                labels = dict((k8s.get("labels") or {}))
                k8s_info = KubernetesInfo(
                    node_name=k8s.get("node_name", name),
                    labels=labels,
                    allocatable_cpu=float(k8s.get("allocatable_cpu", 0.0)),
                    allocatable_mem_gb=float(k8s.get("allocatable_mem_gb", 0.0)),
                    zone=k8s.get("zone"),
                )
            network = None
            if network_info:
                try:
                    network = NetworkIdentity(**network_info)
                except TypeError:
                    network = None
            pool_key = pool_info.get("key")
            pool_score = float(pool_info.get("score", 0.0)) if pool_info else 0.0
            node_obj = Node(
                name=name,
                hardware=hw,
                runtime=rt,
                k8s=k8s_info,
                available=not (node_dict.get("dyn") or {}).get("down", False),
                network=network,
                pool_key=pool_key,
                pool_score=pool_score,
            )
            return node_obj
        except Exception:
            return None

    def _apply_virtualization_locked(self, name: str, node: Dict[str, Any]) -> None:
        k8s = node.get("k8s") or {}
        labels = dict((k8s.get("labels") or {}))
        cluster_name = labels.get("dt.cluster.name")
        if not cluster_name:
            return

        cluster_type = labels.get("dt.cluster.type", "unknown")
        cluster_info = self.clusters.get(cluster_name)
        if not cluster_info:
            cluster_info = ClusterInfo(name=cluster_name, cluster_type=cluster_type or "unknown")

        self._virtual.register_cluster(cluster_info)

        node_dict = copy.deepcopy(node)
        node_dict.setdefault("k8s", {})["labels"] = labels
        node_dict.pop("pool", None)
        node_dict.pop("network", None)
        node_obj = self._node_from_dict_locked(name, node_dict)
        if not node_obj:
            self.clusters[cluster_name] = cluster_info
            return

        identity, pool = self._virtual.ensure_node(cluster_name, node_obj)
        if identity:
            node["network"] = asdict(identity)
            labels["dt.virtual.cluster_id"] = str(identity.cluster_id)
            labels["dt.virtual.node_id"] = str(identity.node_id)
            labels["dt.virtual.mac"] = identity.mac_address
            cluster_info.cluster_id = identity.cluster_id
            cluster_info.network_cidr = identity.network_cidr
            cluster_info.pod_cidr = identity.pod_cidr
            cluster_info.service_cidr = identity.service_cidr

        if pool:
            node.setdefault("pool", {})
            node["pool"].update({
                "key": node_obj.pool_key,
                "score": node_obj.pool_score,
                "characteristics": dict(pool.characteristics),
            })
            cluster_info.node_pools.setdefault(pool.pool_key, [])
            if name not in cluster_info.node_pools[pool.pool_key]:
                cluster_info.node_pools[pool.pool_key].append(name)

        if node_obj.pool_key:
            labels["dt.node.pool"] = node_obj.pool_key

        node.setdefault("k8s", {})["labels"] = labels

        cluster_nodes = set(cluster_info.nodes)
        cluster_nodes.add(name)
        cluster_info.nodes = sorted(cluster_nodes)
        self.clusters[cluster_name] = cluster_info

    def _refresh_virtual_topology_from_nodes_locked(self) -> None:
        self._virtual = VirtualTopology()
        for cluster_name, info in list(self.clusters.items()):
            info.node_pools = {}
            info.nodes = []
            self._virtual.register_cluster(info)
            self.clusters[cluster_name] = info

        for name, node in self.nodes_by_name.items():
            node.pop("network", None)
            node.pop("pool", None)
            self._apply_virtualization_locked(name, node)
    
    def mark_node_availability(self, node_name: str, available: bool) -> None:
        """Mark a node as available or unavailable (for full DTState implementation)."""
        with self._lock:
            if node_name in self.nodes_by_name:
                node_dict = self.nodes_by_name[node_name]
                # Update availability in the dict structure
                if "dyn" not in node_dict:
                    node_dict["dyn"] = {}
                node_dict["dyn"]["available"] = available
                # Also update in the effective state
                if "effective" not in node_dict:
                    node_dict["effective"] = {}
                node_dict["effective"]["available"] = available
                # Invalidate snapshot cache
                self._invalidate_snapshot_locked()

    def register_cluster(self, cluster_info: ClusterInfo) -> None:
        """Register cluster metadata for compatibility with the planner."""
        with self._lock:
            self._virtual.register_cluster(cluster_info)
            self.clusters[cluster_info.name] = cluster_info

    def get_cluster(self, node_name: str) -> Optional[str]:
        """Return the cluster name for a given node, if available."""
        with self._lock:
            node_dict = self.nodes_by_name.get(node_name, {})
            labels = (node_dict.get("k8s") or {}).get("labels", {})
            cluster_label = labels.get("dt.cluster.name")
            if cluster_label:
                return cluster_label

            for cluster_name, cluster_info in self.clusters.items():
                if node_name in cluster_info.nodes:
                    return cluster_name

            for cluster_name in self.clusters.keys():
                if node_name.startswith(cluster_name):
                    return cluster_name

            return None

    def list_nodes(self) -> List[Node]:
        """Return list of Node objects from the state."""
        with self._lock:
            nodes = []
            for name, node_dict in self.nodes_by_name.items():
                node = self._node_from_dict_locked(name, node_dict)
                if node:
                    nodes.append(node)
            return nodes

    def upsert_node(self, node: Node) -> None:
        """Add or update a node from a Node dataclass object."""
        cluster_name = None
        cluster_labels: Dict[str, str] = {}
        if node.k8s:
            cluster_labels = dict(node.k8s.labels or {})
            cluster_name = cluster_labels.get("dt.cluster.name")

        if cluster_name:
            cluster_info = self.clusters.get(cluster_name)
            if not cluster_info:
                cluster_type = cluster_labels.get("dt.cluster.type", "unknown")
                cluster_info = ClusterInfo(name=cluster_name, cluster_type=cluster_type)
            self._virtual.register_cluster(cluster_info)
            identity, pool = self._virtual.ensure_node(cluster_name, node)
            if identity:
                node.network = identity
                if node.k8s:
                    node.k8s.labels = cluster_labels
                    node.k8s.labels.setdefault("dt.cluster.name", cluster_name)
                    node.k8s.labels["dt.virtual.cluster_id"] = str(identity.cluster_id)
                    node.k8s.labels["dt.virtual.node_id"] = str(identity.node_id)
                    node.k8s.labels["dt.virtual.mac"] = identity.mac_address
                cluster_info.cluster_id = identity.cluster_id
                cluster_info.network_cidr = identity.network_cidr
                cluster_info.pod_cidr = identity.pod_cidr
                cluster_info.service_cidr = identity.service_cidr
            if pool:
                cluster_info.node_pools.setdefault(pool.pool_key, [])
                if node.name not in cluster_info.node_pools[pool.pool_key]:
                    cluster_info.node_pools[pool.pool_key].append(node.name)
            if node.pool_key and node.k8s:
                node.k8s.labels["dt.node.pool"] = node.pool_key
            if node.name not in cluster_info.nodes:
                cluster_info.nodes.append(node.name)
            self.clusters[cluster_name] = cluster_info

        descriptor = {
            "name": node.name,
            "hardware": {
                "cpu_cores": node.hardware.cpu_cores,
                "base_ghz": node.hardware.base_ghz,
                "memory_gb": node.hardware.memory_gb,
                "gpu_vram_gb": node.hardware.gpu_vram_gb,
                "arch": node.hardware.arch,
                "tdp_w": node.hardware.tdp_w,
                "memory_mhz": node.hardware.memory_mhz,
                "memory_type": node.hardware.memory_type,
                "accelerators": list(node.hardware.accelerators),
            },
            "runtime": {
                "native_formats": node.runtime.native_formats,
                "emulation_support": node.runtime.emulation_support,
                "wasm_support": node.runtime.wasm_support,
            },
            "k8s": {
                "node_name": node.k8s.node_name,
                "labels": node.k8s.labels,
                "allocatable_cpu": node.k8s.allocatable_cpu,
                "allocatable_mem_gb": node.k8s.allocatable_mem_gb,
                "zone": node.k8s.zone,
            } if node.k8s else {},
        }
        if node.network:
            descriptor["network"] = asdict(node.network)
        if node.pool_key:
            descriptor["pool"] = {
                "key": node.pool_key,
                "score": node.pool_score,
            }
        self.add_or_update_node(descriptor, persist=False, preserve_runtime=True)

    def add_job(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.name] = job

    def get_job(self, job_name: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_name)

    def record_observed_metrics(self, plan_id: str, metrics: ObservedMetrics) -> None:
        with self._lock:
            self.observed_metrics[plan_id] = metrics

    def get_observed_metrics(self, plan_id: str) -> Optional[ObservedMetrics]:
        with self._lock:
            return self.observed_metrics.get(plan_id)

    def list_node_pools(self) -> List[NodePool]:
        with self._lock:
            return self._virtual.list_pools()

    def get_virtual_identity(self, node_name: str) -> Optional[NetworkIdentity]:
        with self._lock:
            ident = self._virtual.get_identity(node_name)
            if ident:
                return ident
            node = self.nodes_by_name.get(node_name)
            if node and node.get("network"):
                try:
                    return NetworkIdentity(**node["network"])
                except TypeError:
                    return None
            return None

    def describe_virtual_topology(self) -> Dict[str, Any]:
        with self._lock:
            return self._virtual.describe()

    def add_or_update_node(
        self,
        descriptor: Dict[str, Any],
        *,
        persist: bool = True,
        preserve_runtime: bool = True,
    ) -> Dict[str, Any]:
        """Insert or update a node descriptor at runtime.

        Args:
            descriptor: Full node descriptor (same shape as YAML on disk).
            persist:   If True, write descriptor back to ``nodes/<name>.yaml``.
            preserve_runtime: Keep existing dyn/reservation data when updating.

        Returns:
            The effective node dictionary stored in ``nodes_by_name``.
        """

        name = (descriptor or {}).get("name")
        if not name:
            raise ValueError("descriptor.name is required")

        disk_descriptor = copy.deepcopy(descriptor)
        disk_descriptor.pop("dyn", None)

        with self._lock:
            prev = self.nodes_by_name.get(name)
            dyn_defaults = NodeDyn().__dict__.copy()

            incoming_dyn = dict(descriptor.get("dyn") or {})
            for key, value in incoming_dyn.items():
                if key in dyn_defaults:
                    if key == "reservations" and isinstance(value, dict):
                        dyn_defaults[key] = dict(value)
                    else:
                        dyn_defaults[key] = value

            if preserve_runtime and prev:
                prev_dyn = prev.get("dyn") or {}
                for key, value in prev_dyn.items():
                    if key == "reservations" and isinstance(value, dict):
                        dyn_defaults[key] = dict(value)
                    elif key in dyn_defaults:
                        dyn_defaults[key] = value

            node = copy.deepcopy(descriptor)
            node["name"] = name
            node["dyn"] = dyn_defaults

            self._compute_and_cache_capacities(node)

            health = node.get("health") or {}
            lifecycle = node.get("lifecycle") or {}
            power = node.get("power") or {}

            self._predictor.ensure_node(
                name,
                reliability=None
                if health.get("reliability") is None
                else safe_float(health.get("reliability"), 0.95),
                availability_window_sec=None
                if lifecycle.get("availability_window_sec") is None
                else safe_float(lifecycle.get("availability_window_sec"), 0.0),
                battery_pct=None
                if power.get("battery_pct") is None
                else safe_float(power.get("battery_pct"), 0.0),
                battery_drain_pct_per_hr=None
                if power.get("battery_drain_pct_per_hr") is None
                else safe_float(power.get("battery_drain_pct_per_hr"), 0.0),
                mtbf_hours=None
                if health.get("mtbf_hours") is None
                else safe_float(health.get("mtbf_hours"), 0.0),
                uptime_hours=None
                if health.get("uptime_hours") is None
                else safe_float(health.get("uptime_hours"), 0.0),
            )

            self._apply_virtualization_locked(name, node)

            self.nodes_by_name[name] = node
            self._nodes_fingerprint[f"{name}.yaml"] = time.time()
            self._update_predictive_for_node_locked(name)
            self._invalidate_snapshot_locked()
            self._emit_event(
                "fabric.node.added",
                {
                    "node": name,
                    "persisted": bool(persist),
                    "preserve_runtime": bool(preserve_runtime),
                },
                subject=name,
            )

        if persist:
            target = self.nodes_dir / f"{name}.yaml"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(yaml.safe_dump(disk_descriptor, sort_keys=False), encoding="utf-8")
                with self._lock:
                    try:
                        self._nodes_fingerprint[target.name] = target.stat().st_mtime
                        self._nodes_mtime = max(self._nodes_mtime, self._nodes_fingerprint[target.name])
                    except FileNotFoundError:
                        pass
            except Exception as exc:
                print(f"[state] WARN: failed to persist node {name}: {exc}")

        return self.get_node(name) or {}

    def node_headroom(self, name: str) -> Optional[Dict[str, float]]:
        """Return instantaneous capacity/free headroom metrics for a node."""

        with self._lock:
            node = self.nodes_by_name.get(name)
            if not node:
                return None
            eff = self._effective_caps(node)
            return {
                "max_cpu_cores": eff.get("max_cpu_cores", 0.0),
                "max_mem_gb": eff.get("max_mem_gb", 0.0),
                "max_gpu_vram_gb": eff.get("max_gpu_vram_gb", 0.0),
                "free_cpu_cores": eff.get("free_cpu_cores", 0.0),
                "free_mem_gb": eff.get("free_mem_gb", 0.0),
                "free_gpu_vram_gb": eff.get("free_gpu_vram_gb", 0.0),
            }

    def reservations_view(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return a copy of all reservations grouped by node."""

        with self._lock:
            out: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for name, node in self.nodes_by_name.items():
                dyn = node.get("dyn") or {}
                reservations = dyn.get("reservations") or {}
                if not reservations:
                    continue
                out[name] = {rid: dict(info) for rid, info in reservations.items()}
            return out

    # -------- public API (write/update) --------

    def apply_observation(self, payload: Dict[str, Any]) -> None:
        """
        Merge an observation (same shape chaos uses):
        { "action": "apply"|"revert", "payload": {"type": "node"|"link", ...}}
        """
        with self._lock:
            p = payload.get("payload", {})
            typ = p.get("type")
            if typ == "node":
                node = p.get("node")
                changes = p.get("changes") or {}
                target = self.nodes_by_name.get(node)
                if not target:
                    return
                dyn = target.setdefault("dyn", NodeDyn().__dict__.copy())
                for k, v in changes.items():
                    if k in dyn:
                        dyn[k] = v
                self._update_predictive_for_node_locked(node)
                self._emit_event("fabric.node.observe", {"node": node, "changes": changes}, subject=node)
            elif typ == "link":
                k = p.get("key")
                changes = p.get("changes") or {}
                link = self.links_by_key.get(k)
                if not link:
                    # Create on the fly if key is valid
                    parts = k.split("|", 1)
                    if len(parts) == 2:
                        link = {"a": parts[0], "b": parts[1], "base": {}, "dyn": LinkDyn().__dict__.copy()}
                        self.links_by_key[k] = link
                    else:
                        return
                dyn = link.setdefault("dyn", LinkDyn().__dict__.copy())
                for kk, vv in changes.items():
                    if kk in dyn:
                        dyn[kk] = vv
                self._update_link_predictive_locked(k)
                self._emit_event("fabric.link.observe", {"link": k, "changes": changes}, subject=k)
        self._invalidate_snapshot_locked()

    # -------- federation + planner helpers --------

    def _derive_federation_name(self, node: Dict[str, Any]) -> str:
        labels = node.get("labels") or {}
        for key in ("federation", "zone", "site", "rack", "region"):
            val = labels.get(key)
            if isinstance(val, str) and val:
                return val
        return "global"

    def _federation_overview_locked(
        self,
        nodes_view: Optional[Dict[str, Dict[str, Any]]] = None,
        links_view: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
        nodes_map = nodes_view if nodes_view is not None else self.nodes_by_name
        links_map = links_view if links_view is not None else self.links_by_key

        stats: Dict[str, Dict[str, Any]] = {}
        node_to_fed: Dict[str, str] = {}

        for name, node in nodes_map.items():
            fed = self._derive_federation_name(node)
            node_to_fed[name] = fed
            entry = stats.setdefault(
                fed,
                {
                    "name": fed,
                    "nodes": [],
                    "total_cpu_cores": 0.0,
                    "free_cpu_cores": 0.0,
                    "total_mem_gb": 0.0,
                    "free_mem_gb": 0.0,
                    "total_gpu_vram_gb": 0.0,
                    "free_gpu_vram_gb": 0.0,
                    "down_nodes": 0,
                    "hot_nodes": 0,
                    "reservations": 0,
                    "trust_sum": 0.0,
                    "trust_count": 0,
                    "loss_sum": 0.0,
                    "loss_count": 0,
                },
            )

            entry["nodes"].append(name)

            eff = self._effective_caps(node)
            caps = node.get("caps", {})
            dyn = node.get("dyn", {})
            labels = node.get("labels", {})

            entry["total_cpu_cores"] += safe_float(caps.get("max_cpu_cores"), 0.0)
            entry["free_cpu_cores"] += safe_float(eff.get("free_cpu_cores"), 0.0)
            entry["total_mem_gb"] += safe_float(caps.get("ram_gb"), 0.0)
            entry["free_mem_gb"] += safe_float(eff.get("free_mem_gb"), 0.0)
            entry["total_gpu_vram_gb"] += safe_float(caps.get("gpu_vram_gb"), 0.0)
            entry["free_gpu_vram_gb"] += safe_float(eff.get("free_gpu_vram_gb"), 0.0)

            if dyn.get("down"):
                entry["down_nodes"] += 1
            if safe_float(dyn.get("thermal_derate"), 0.0) >= 0.25:
                entry["hot_nodes"] += 1

            reservations = dyn.get("reservations") or {}
            entry["reservations"] += len(reservations)

            trust = labels.get("trust")
            try:
                if trust is not None:
                    tval = float(trust)
                    entry["trust_sum"] += tval
                    entry["trust_count"] += 1
            except Exception:
                pass

            loss_pct = safe_float((node.get("network") or {}).get("loss_pct"), None)
            if loss_pct is not None:
                entry["loss_sum"] += loss_pct
                entry["loss_count"] += 1

        federations: List[Dict[str, Any]] = []
        for fed, entry in stats.items():
            total_cpu = entry["total_cpu_cores"] or 0.0
            free_cpu = entry["free_cpu_cores"] or 0.0
            total_nodes = len(entry["nodes"])
            trust_avg = (
                entry["trust_sum"] / max(1, entry["trust_count"])
                if entry["trust_count"]
                else None
            )
            loss_avg = (
                entry["loss_sum"] / max(1, entry["loss_count"])
                if entry["loss_count"]
                else None
            )

            federations.append(
                {
                    "name": fed,
                    "nodes": list(entry["nodes"]),
                    "total_cpu_cores": round(total_cpu, 4),
                    "free_cpu_cores": round(free_cpu, 4),
                    "total_mem_gb": round(entry["total_mem_gb"], 4),
                    "free_mem_gb": round(entry["free_mem_gb"], 4),
                    "total_gpu_vram_gb": round(entry["total_gpu_vram_gb"], 4),
                    "free_gpu_vram_gb": round(entry["free_gpu_vram_gb"], 4),
                    "down_nodes": entry["down_nodes"],
                    "hot_nodes": entry["hot_nodes"],
                    "reservations": entry["reservations"],
                    "avg_trust": round(trust_avg, 4) if trust_avg is not None else None,
                    "avg_loss_pct": round(loss_avg, 4) if loss_avg is not None else None,
                    "load_factor": 0.0
                    if total_cpu <= 0
                    else clamp(
                        (total_cpu - free_cpu) / max(1e-6, total_cpu), 0.0, 1.0
                    ),
                    "down_fraction": 0.0
                    if total_nodes == 0
                    else round(entry["down_nodes"] / total_nodes, 4),
                    "hot_fraction": 0.0
                    if total_nodes == 0
                    else round(entry["hot_nodes"] / total_nodes, 4),
                }
            )

        # Aggregate cross-federation link health (best effort)
        link_buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for key, link in links_map.items():
            a = link.get("a")
            b = link.get("b")
            if not a or not b:
                continue
            fa = node_to_fed.get(a, a)
            fb = node_to_fed.get(b, b)
            if fa == fb:
                continue
            pair = tuple(sorted((fa, fb)))
            bucket = link_buckets.setdefault(
                pair,
                {
                    "a": pair[0],
                    "b": pair[1],
                    "links": 0,
                    "down": 0,
                    "min_speed_gbps": float("inf"),
                    "max_loss_pct": 0.0,
                    "avg_rtt_ms_sum": 0.0,
                },
            )
            eff = self._effective_link(link)
            bucket["links"] += 1
            if eff.get("down"):
                bucket["down"] += 1
            spd = safe_float(eff.get("speed_gbps"), float("inf"))
            bucket["min_speed_gbps"] = min(bucket["min_speed_gbps"], spd)
            bucket["max_loss_pct"] = max(
                bucket["max_loss_pct"], safe_float(eff.get("loss_pct"), 0.0)
            )
            bucket["avg_rtt_ms_sum"] += safe_float(eff.get("rtt_ms"), 0.0)

        federation_links: List[Dict[str, Any]] = []
        for pair, bucket in link_buckets.items():
            links_count = bucket["links"] or 1
            min_speed = bucket["min_speed_gbps"]
            if min_speed == float("inf"):
                min_speed = None
            federation_links.append(
                {
                    "a": bucket["a"],
                    "b": bucket["b"],
                    "links": bucket["links"],
                    "down_links": bucket["down"],
                    "min_speed_gbps": None if min_speed is None else round(min_speed, 4),
                    "max_loss_pct": round(bucket["max_loss_pct"], 4),
                    "avg_rtt_ms": round(bucket["avg_rtt_ms_sum"] / links_count, 4),
                }
            )

        federations.sort(key=lambda x: x["name"])
        federation_links.sort(key=lambda x: (x["a"], x["b"]))

        return federations, federation_links, node_to_fed

    def nodes_for_planner(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            out: Dict[str, Dict[str, Any]] = {}
            for name, node in self.nodes_by_name.items():
                cp = copy.deepcopy(node)
                cp["effective"] = self._effective_caps(node)
                forecast = self._predictor.node_forecast(name)
                cp["predictive"] = {
                    "util_now": forecast.util_now,
                    "util_forecast": forecast.util_forecast,
                    "reliability": forecast.reliability,
                    "availability_window_sec": forecast.availability_window_sec,
                    "projected_derate": forecast.projected_derate,
                }
                out[name] = cp
            return out

    def federations_overview(self) -> Dict[str, Any]:
        with self._lock:
            federations, federation_links, node_federations = self._federation_overview_locked()
            return {
                "federations": federations,
                "federation_links": federation_links,
                "node_federations": node_federations,
            }

    def federation_stats(self) -> Dict[str, Any]:
        return self.federations_overview()

    def federation_for_node(self, node_name: str) -> Optional[str]:
        with self._lock:
            node = self.nodes_by_name.get(node_name)
            if not node:
                return None
            return self._derive_federation_name(node)

    def _effective_link_between_locked(self, a: str, b: str) -> Dict[str, Any]:
        k = link_key(a, b)
        link = self.links_by_key.get(k)
        if link:
            eff = self._effective_link(link)
            eff["estimated"] = False
            return eff

        # Fallback estimation using node network hints and defaults
        na = self.nodes_by_name.get(a)
        nb = self.nodes_by_name.get(b)
        netdef = self.defaults.get("network", {}) or {}

        def _node_speed(node: Optional[Dict[str, Any]]) -> float:
            if not node:
                return safe_float(netdef.get("speed_gbps"), 1.0)
            net = node.get("network") or {}
            spd = safe_float(net.get("speed_gbps"), None)
            if spd is None:
                bw = safe_float(net.get("base_bandwidth_mbps"), 0.0)
                if bw > 0:
                    spd = bw / 1000.0
            return spd if spd is not None else safe_float(netdef.get("speed_gbps"), 1.0)

        eff_speed = min(_node_speed(na), _node_speed(nb))
        eff_rtt = safe_float(netdef.get("rtt_ms"), 5.0)
        eff_loss = safe_float(netdef.get("loss_pct"), 0.0)
        eff_jitter = safe_float(netdef.get("jitter_ms"), 0.5)

        if na:
            eff_rtt = max(eff_rtt, safe_float((na.get("network") or {}).get("base_latency_ms"), eff_rtt))
            eff_loss = max(eff_loss, safe_float((na.get("network") or {}).get("loss_pct"), eff_loss))
        if nb:
            eff_rtt = max(eff_rtt, safe_float((nb.get("network") or {}).get("base_latency_ms"), eff_rtt))
            eff_loss = max(eff_loss, safe_float((nb.get("network") or {}).get("loss_pct"), eff_loss))

        return {
            "estimated": True,
            "down": False,
            "speed_gbps": eff_speed,
            "rtt_ms": eff_rtt,
            "jitter_ms": eff_jitter,
            "loss_pct": eff_loss,
        }

    def effective_link_between(self, a: Optional[str], b: str) -> Dict[str, Any]:
        if not a or a == b:
            return {"estimated": True, "down": False, "speed_gbps": float("inf"), "rtt_ms": 0.0, "jitter_ms": 0.0, "loss_pct": 0.0}
        with self._lock:
            return self._effective_link_between_locked(a, b)

    # -------- reservations --------

    def reserve(self, req: Dict[str, Any]) -> Optional[str]:
        """
        Try to reserve resources on a specific node or choose one automatically.

        req example:
        {
          "node": "ws-001",                # optional; if omitted, caller should choose a node via planner
          "cpu_cores": 2.0,
          "mem_gb": 4.0,
          "gpu_vram_gb": 2.0
        }
        """
        with self._lock:
            node_name = req.get("node")
            if not node_name:
                return None
            n = self.nodes_by_name.get(node_name)
            if not n:
                return None
            if self._is_down(n):
                return None

            eff = self._effective_caps(n)
            need_cpu = safe_float(req.get("cpu_cores"), 0.0)
            need_mem = safe_float(req.get("mem_gb"), 0.0)
            need_vram = safe_float(req.get("gpu_vram_gb"), 0.0)

            if eff["free_cpu_cores"] + 1e-9 < need_cpu:
                return None
            if eff["free_mem_gb"] + 1e-9 < need_mem:
                return None
            if eff["free_gpu_vram_gb"] + 1e-9 < need_vram:
                return None

            # allocate
            dyn = n.setdefault("dyn", NodeDyn().__dict__.copy())
            dyn["used_cpu_cores"] += need_cpu
            dyn["used_mem_gb"] += need_mem
            dyn["used_gpu_vram_gb"] += need_vram

            rid = f"res-{self._res_seq:07d}"
            self._res_seq += 1
            dyn.setdefault("reservations", {})[rid] = {
                "cpu_cores": need_cpu,
                "mem_gb": need_mem,
                "gpu_vram_gb": need_vram,
                "ts": utc_ms(),
            }
            forecast = self._update_predictive_for_node_locked(node_name)
            self._invalidate_snapshot_locked()
            self._emit_event(
                "fabric.reservation.created",
                {
                    "node": node_name,
                    "reservation_id": rid,
                    "cpu_cores": need_cpu,
                    "mem_gb": need_mem,
                    "gpu_vram_gb": need_vram,
                    "forecast": None
                    if forecast is None
                    else {
                        "util_now": forecast.util_now,
                        "util_forecast": forecast.util_forecast,
                        "reliability": forecast.reliability,
                    },
                },
                subject=node_name,
            )
            return rid

    def release(self, node_name: str, reservation_id: str) -> bool:
        with self._lock:
            n = self.nodes_by_name.get(node_name)
            if not n:
                return False
            dyn = n.get("dyn") or {}
            res = (dyn.get("reservations") or {}).pop(reservation_id, None)
            if not res:
                return False
            dyn["used_cpu_cores"] = max(0.0, dyn.get("used_cpu_cores", 0.0) - safe_float(res.get("cpu_cores"), 0.0))
            dyn["used_mem_gb"] = max(0.0, dyn.get("used_mem_gb", 0.0) - safe_float(res.get("mem_gb"), 0.0))
            dyn["used_gpu_vram_gb"] = max(0.0, dyn.get("used_gpu_vram_gb", 0.0) - safe_float(res.get("gpu_vram_gb"), 0.0))
            forecast = self._update_predictive_for_node_locked(node_name)
            self._invalidate_snapshot_locked()
            self._emit_event(
                "fabric.reservation.released",
                {
                    "node": node_name,
                    "reservation_id": reservation_id,
                    "forecast": None
                    if forecast is None
                    else {
                        "util_now": forecast.util_now,
                        "util_forecast": forecast.util_forecast,
                        "reliability": forecast.reliability,
                    },
                },
                subject=node_name,
            )
            return True

    def recent_events(self, limit: int = 100, since_id: Optional[str] = None) -> List[Dict[str, Any]]:
        events = self._events.recent(limit=limit, since_id=since_id)
        return [dict(evt) for evt in events]

    def predictive_overview(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._predictor.overview())

    def node_reliability(self, node_name: str) -> float:
        with self._lock:
            return self._predictor.node_forecast(node_name).reliability

    def predict_node_derate(self, node_name: str) -> float:
        with self._lock:
            return self._predictor.node_forecast(node_name).projected_derate

    def node_availability_window(self, node_name: str) -> Optional[float]:
        with self._lock:
            return self._predictor.node_forecast(node_name).availability_window_sec

    def link_variability(self, a: str, b: str) -> Dict[str, Any]:
        key = link_key(a, b)
        with self._lock:
            forecast = self._predictor.link_forecast(key)
        return {
            "latency_ms": forecast.latency_ms,
            "jitter_ms": forecast.jitter_ms,
            "loss_pct": forecast.loss_pct,
            "latency_p95_ms": forecast.latency_p95_ms,
        }

    # -------- scoring utility (baseline) --------

    def score_node_basic(self, stage: Dict[str, Any], node: Dict[str, Any]) -> float:
        """
        Lower is better. Very simple latency proxy:
        - Penalize 'down', thermal_derate, low CPU_units
        - Give a boost if formats_supported matches stage's allowed_formats (cuda/npu)
        """
        if self._is_down(node):
            return 1e12

        caps = node.get("caps", {})
        dyn = node.get("dyn", {})
        cpu_units = safe_float(caps.get("cpu_units"), 0.0)
        derate = safe_float(dyn.get("thermal_derate"), 0.0)

        # format preference
        allowed = set(stage.get("allowed_formats") or [])
        fmts = set(node.get("formats_supported") or [])
        fmt_bonus = 0.0
        if allowed:
            if fmts & allowed:
                fmt_bonus = -0.15  # reduce score (better)
            else:
                fmt_bonus = +0.25  # increase score (worse)

        score = (1.0 / max(1e-6, cpu_units)) * (1.0 + derate) * (1.0 + fmt_bonus)
        return max(0.0, score)

    # -------- effective capacities/links --------

    def _is_down(self, node: Dict[str, Any]) -> bool:
        dyn = node.get("dyn") or {}
        return bool(dyn.get("down", False))

    def _effective_caps(self, node: Dict[str, Any]) -> Dict[str, float]:
        caps = node.get("caps", {})
        dyn = node.get("dyn", {})
        derate = safe_float(dyn.get("thermal_derate"), 0.0)

        max_cpu = safe_float(caps.get("max_cpu_cores"), 0.0)
        max_mem = safe_float(caps.get("ram_gb"), 0.0)
        max_vram = safe_float(caps.get("gpu_vram_gb"), 0.0)

        # Thermal derate reduces effective usable CPU (you can make this fancier later)
        eff_cpu = max_cpu * (1.0 - max(0.0, min(1.0, derate)))

        used_cpu = safe_float(dyn.get("used_cpu_cores"), 0.0)
        used_mem = safe_float(dyn.get("used_mem_gb"), 0.0)
        used_vram = safe_float(dyn.get("used_gpu_vram_gb"), 0.0)

        return {
            "max_cpu_cores": max_cpu,
            "max_mem_gb": max_mem,
            "max_gpu_vram_gb": max_vram,
            "free_cpu_cores": max(0.0, eff_cpu - used_cpu),
            "free_mem_gb": max(0.0, max_mem - used_mem),
            "free_gpu_vram_gb": max(0.0, max_vram - used_vram),
        }

    def _effective_link(self, link: Dict[str, Any]) -> Dict[str, Any]:
        base = link.get("base", {}) or {}
        dyn = link.get("dyn", {}) or {}

        # Choose dyn override if set, else base, else topology defaults
        def pick(key: str, default_key: Optional[str] = None, default_val: Optional[Any] = None):
            if key in dyn and dyn[key] is not None:
                return dyn[key]
            if key in base and base[key] is not None:
                return base[key]
            if default_key:
                # Look up defaults.network
                netdef = (self.defaults.get("network") or {})
                return netdef.get(default_key, default_val)
            return default_val

        eff = {
            "down": bool(dyn.get("down", False)),
            "speed_gbps": safe_float(pick("speed_gbps", "speed_gbps", 1.0), 1.0),
            "rtt_ms": safe_float(pick("rtt_ms", "rtt_ms", 5.0), 5.0),
            "jitter_ms": safe_float(pick("jitter_ms", "jitter_ms", 0.5), 0.5),
            "loss_pct": safe_float(pick("loss_pct", "loss_pct", 0.0), 0.0),
            "ecn": bool(pick("ecn", "ecn", False)),
        }
        return eff

    # -------- disk persistence for overrides (optional) --------

    def write_overrides(self) -> None:
        """Persist current dyn states to sim/overrides.json (lossy for unknown fields)."""
        with self._lock:
            out = {"nodes": {}, "links": {}}
            for name, n in self.nodes_by_name.items():
                dyn = n.get("dyn") or {}
                # Only write meaningful keys
                nd = {}
                for k in ("down", "power_cap_w", "thermal_derate", "clock_skew_ms",
                          "packet_dup", "packet_reorder"):
                    if k in dyn and dyn[k] not in (None, False, 0, 0.0):
                        nd[k] = dyn[k]
                if nd:
                    out["nodes"][name] = nd

            for k, l in self.links_by_key.items():
                dyn = l.get("dyn") or {}
                ld = {}
                for kk in ("down", "speed_gbps", "rtt_ms", "jitter_ms", "loss_pct", "ecn"):
                    if kk in dyn and dyn[kk] not in (None, False, 0, 0.0):
                        ld[kk] = dyn[kk]
                if ld:
                    out["links"][k] = ld

            try:
                self.overrides_path.parent.mkdir(parents=True, exist_ok=True)
                self.overrides_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
                self._overrides = out
                self._overrides_mtime = self.overrides_path.stat().st_mtime
            except Exception as e:
                print(f"[state] WARN: failed to write overrides: {e}")


# ----------------------------- manual test -----------------------------

if __name__ == "__main__":
    st = DTState(auto_start_watchers=False)  # don't spawn watcher for a one-off test
    snap = st.snapshot()
    print(f"Loaded nodes: {len(snap['nodes'])}, links: {len(snap['links'])}")

    # Reserve a tiny slice on the first node (if any)
    if snap["nodes"]:
        n0 = snap["nodes"][0]["name"]
        rid = st.reserve({"node": n0, "cpu_cores": 1, "mem_gb": 2})
        print("Reservation:", n0, rid)
        if rid:
            st.release(n0, rid)
            print("Released:", rid)

