"""Telemetry collector using Kubernetes Metrics API and watch API."""

from __future__ import annotations

import time
import threading
import logging
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

from kubernetes import client, config, watch
from kubernetes.client.exceptions import ApiException

from dt.state import DTState, ObservedMetrics

logger = logging.getLogger(__name__)


@dataclass
class PodEvent:
	"""Pod lifecycle event."""
	name: str
	namespace: str
	phase: str  # Pending, Running, Succeeded, Failed
	labels: Dict[str, str]
	start_time: Optional[float] = None
	completion_time: Optional[float] = None
	node_name: Optional[str] = None


class TelemetryCollector:
	"""
	Collects telemetry from Kubernetes clusters.
	
	Uses:
	- Kubernetes Metrics API for CPU/memory metrics
	- Watch API for pod/node lifecycle events
	- cAdvisor (via metrics API) for container-level metrics
	"""
	
	def __init__(
		self,
		state: DTState,
		cluster_manager: Optional[Any] = None,  # ClusterManager type
		namespace: str = "dt-fabric",
		update_interval_s: float = 5.0,
	) -> None:
		"""
		Initialize telemetry collector.
		
		Args:
			state: DTState to update with telemetry
			cluster_manager: ClusterManager instance (optional)
			namespace: Kubernetes namespace to watch
			update_interval_s: Interval for metrics collection (seconds)
		"""
		self.state = state
		self.cluster_manager = cluster_manager
		self.namespace = namespace
		self.update_interval_s = update_interval_s
		
		self._running = False
		self._threads: list[threading.Thread] = []
		self._stop_event = threading.Event()
		
		# Callbacks
		self.on_pod_complete: Optional[Callable[[str, ObservedMetrics], None]] = None
		
		# Track pod start times for latency calculation
		self._pod_start_times: Dict[str, float] = {}  # pod_name -> start_time
		
		logger.info(f"TelemetryCollector initialized for namespace: {namespace}")
	
	def start(self) -> None:
		"""Start collecting telemetry."""
		if self._running:
			logger.warning("TelemetryCollector already running")
			return
		
		self._running = True
		self._stop_event.clear()
		
		# Start metrics collection thread
		metrics_thread = threading.Thread(
			target=self._collect_metrics_loop,
			name="telemetry-metrics",
			daemon=True
		)
		metrics_thread.start()
		self._threads.append(metrics_thread)
		
		# Start pod watch thread
		pod_thread = threading.Thread(
			target=self._watch_pods_loop,
			name="telemetry-pods",
			daemon=True
		)
		pod_thread.start()
		self._threads.append(pod_thread)
		
		# Start node watch thread
		node_thread = threading.Thread(
			target=self._watch_nodes_loop,
			name="telemetry-nodes",
			daemon=True
		)
		node_thread.start()
		self._threads.append(node_thread)
		
		logger.info("TelemetryCollector started")
	
	def stop(self) -> None:
		"""Stop collecting telemetry."""
		if not self._running:
			return
		
		self._running = False
		self._stop_event.set()
		
		# Wait for threads to finish
		for thread in self._threads:
			thread.join(timeout=5.0)
		
		self._threads.clear()
		logger.info("TelemetryCollector stopped")
	
	def _collect_metrics_loop(self) -> None:
		"""Main loop for collecting node/pod metrics."""
		# Try to load kubeconfig
		try:
			config.load_incluster_config()
		except config.ConfigException:
			try:
				config.load_kube_config()
			except Exception as e:
				logger.error(f"Failed to load Kubernetes config: {e}")
				return
		
		metrics_client = client.CustomObjectsApi()
		core_client = client.CoreV1Api()
		
		while not self._stop_event.is_set():
			try:
				# Collect node metrics
				self._collect_node_metrics(core_client, metrics_client)
				
				# Collect pod metrics
				self._collect_pod_metrics(core_client, metrics_client)
				
			except Exception as e:
				logger.error(f"Error collecting metrics: {e}")
			
			# Wait for next interval
			self._stop_event.wait(self.update_interval_s)
	
	def _collect_node_metrics(
		self,
		core_client: client.CoreV1Api,
		metrics_client: client.CustomObjectsApi
	) -> None:
		"""Collect CPU/memory metrics for nodes."""
		try:
			# Get node metrics from metrics API
			# Note: This requires metrics-server to be installed
			try:
				metrics = metrics_client.get_cluster_custom_object(
					group="metrics.k8s.io",
					version="v1beta1",
					name="nodes",
					plural="nodes"
				)
			except ApiException as e:
				if e.status == 404:
					# Metrics API not available, skip
					return
				raise
			
			# Process node metrics
			for item in metrics.get('items', []):
				node_name = item.get('metadata', {}).get('name')
				if not node_name:
					continue
				
				usage = item.get('usage', {})
				cpu_str = usage.get('cpu', '0')
				memory_str = usage.get('memory', '0')
				
				# Parse CPU (e.g., "100m" -> 0.1 cores)
				cpu_cores = self._parse_cpu(cpu_str)
				
				# Parse memory (e.g., "100Mi" -> MB)
				memory_bytes = self._parse_memory(memory_str)
				memory_gb = memory_bytes / (1024 ** 3)
				
				# Get node capacity for utilization calculation
				node = core_client.read_node(node_name)
				capacity = node.status.capacity
				total_cpu = self._parse_cpu(capacity.get('cpu', '0'))
				total_memory = self._parse_memory(capacity.get('memory', '0'))
				
				cpu_util = (cpu_cores / total_cpu * 100.0) if total_cpu > 0 else 0.0
				mem_util = (memory_bytes / total_memory * 100.0) if total_memory > 0 else 0.0
				
				# Update state
				self.state.update_node_telemetry(node_name, {
					'cpu_util': cpu_util,
					'mem_util': mem_util,
				})
		
		except Exception as e:
			logger.debug(f"Error collecting node metrics: {e}")
	
	def _collect_pod_metrics(
		self,
		core_client: client.CoreV1Api,
		metrics_client: client.CustomObjectsApi
	) -> None:
		"""Collect CPU/memory metrics for pods."""
		try:
			# Get pod metrics from metrics API
			try:
				metrics = metrics_client.list_namespaced_custom_object(
					group="metrics.k8s.io",
					version="v1beta1",
					namespace=self.namespace,
					plural="pods"
				)
			except ApiException as e:
				if e.status == 404:
					# Metrics API not available, skip
					return
				raise
			
			# Process pod metrics
			for item in metrics.get('items', []):
				pod_name = item.get('metadata', {}).get('name')
				if not pod_name:
					continue
				
				# Extract plan_id from labels
				labels = item.get('metadata', {}).get('labels', {})
				plan_id = labels.get('dt.plan_id')
				if not plan_id:
					continue
				
				# Sum container metrics
				containers = item.get('containers', [])
				total_cpu = 0.0
				total_memory = 0.0
				
				for container in containers:
					usage = container.get('usage', {})
					total_cpu += self._parse_cpu(usage.get('cpu', '0'))
					total_memory += self._parse_memory(usage.get('memory', '0'))
				
				# Store for later use in verification
				# (we'll aggregate when pod completes)
		
		except Exception as e:
			logger.debug(f"Error collecting pod metrics: {e}")
	
	def _watch_pods_loop(self) -> None:
		"""Watch pod lifecycle events."""
		try:
			config.load_incluster_config()
		except config.ConfigException:
			try:
				config.load_kube_config()
			except Exception as e:
				logger.error(f"Failed to load Kubernetes config: {e}")
				return
		
		core_client = client.CoreV1Api()
		w = watch.Watch()
		
		while not self._stop_event.is_set():
			try:
				for event in w.stream(
					core_client.list_namespaced_pod,
					namespace=self.namespace,
					label_selector="app=dt-worker",
					timeout_seconds=30
				):
					if self._stop_event.is_set():
						break
					
					pod = event['object']
					pod_name = pod.metadata.name
					phase = pod.status.phase
					
					# Extract labels
					labels = pod.metadata.labels or {}
					plan_id = labels.get('dt.plan_id')
					
					# Record event
					pod_event = {
						'name': pod_name,
						'type': phase,
						'labels': labels,
						'node_name': pod.spec.node_name,
					}
					self.state.record_pod_event(pod_event)
					
					# Track start time
					if phase == 'Running' and pod_name not in self._pod_start_times:
						start_time = pod.status.start_time
						if start_time:
							self._pod_start_times[pod_name] = start_time.timestamp()
					
					# Handle completion
					if phase in ('Succeeded', 'Failed'):
						self._handle_pod_completion(
							pod_name,
							plan_id,
							phase,
							pod.status.start_time,
							time.time()
						)
			
			except Exception as e:
				logger.error(f"Error watching pods: {e}")
				time.sleep(5)  # Wait before retrying
	
	def _watch_nodes_loop(self) -> None:
		"""Watch node status changes."""
		try:
			config.load_incluster_config()
		except config.ConfigException:
			try:
				config.load_kube_config()
			except Exception as e:
				logger.error(f"Failed to load Kubernetes config: {e}")
				return
		
		core_client = client.CoreV1Api()
		w = watch.Watch()
		
		while not self._stop_event.is_set():
			try:
				for event in w.stream(
					core_client.list_node,
					timeout_seconds=30
				):
					if self._stop_event.is_set():
						break
					
					node = event['object']
					node_name = node.metadata.name
					
					# Check node conditions
					ready = False
					for condition in node.status.conditions or []:
						if condition.type == 'Ready':
							ready = condition.status == 'True'
							break
					
					# Update availability in state
					self.state.mark_node_availability(node_name, ready)
			
			except Exception as e:
				logger.error(f"Error watching nodes: {e}")
				time.sleep(5)  # Wait before retrying
	
	def _handle_pod_completion(
		self,
		pod_name: str,
		plan_id: Optional[str],
		phase: str,
		start_time: Optional[Any],
		end_time: float
	) -> None:
		"""Handle pod completion and collect final metrics."""
		if not plan_id:
			return
		
		# Calculate latency
		start_ts = self._pod_start_times.pop(pod_name, None)
		if start_time:
			start_ts = start_time.timestamp()
		
		if not start_ts:
			logger.warning(f"Pod {pod_name} completed but no start time recorded")
			return
		
		latency_ms = (end_time - start_ts) * 1000.0
		
		# Create observed metrics
		# Note: CPU/memory peak would be collected from metrics API
		# For now, we use placeholder values
		observed = ObservedMetrics(
			latency_ms=latency_ms,
			cpu_util=0.0,  # TODO: Collect from metrics API
			mem_peak_gb=0.0,  # TODO: Collect from metrics API
			energy_kwh=0.0,  # TODO: Estimate from CPU time
			completed_at=end_time
		)
		
		# Record in state
		self.state.record_observed_metrics(plan_id, observed)
		
		# Trigger callback if set
		if self.on_pod_complete:
			try:
				self.on_pod_complete(plan_id, observed)
			except Exception as e:
				logger.error(f"Error in pod completion callback: {e}")
	
	def _parse_cpu(self, cpu_str: str) -> float:
		"""Parse CPU string (e.g., '100m' -> 0.1, '2' -> 2.0)."""
		if not cpu_str:
			return 0.0
		
		cpu_str = cpu_str.strip()
		if cpu_str.endswith('m'):
			return float(cpu_str[:-1]) / 1000.0
		return float(cpu_str)
	
	def _parse_memory(self, memory_str: str) -> float:
		"""Parse memory string (e.g., '100Mi' -> bytes)."""
		if not memory_str:
			return 0.0
		
		memory_str = memory_str.strip()
		
		# Handle binary units
		if memory_str.endswith('Ki'):
			return float(memory_str[:-2]) * 1024
		elif memory_str.endswith('Mi'):
			return float(memory_str[:-2]) * (1024 ** 2)
		elif memory_str.endswith('Gi'):
			return float(memory_str[:-2]) * (1024 ** 3)
		# Handle decimal units
		elif memory_str.endswith('K'):
			return float(memory_str[:-1]) * 1000
		elif memory_str.endswith('M'):
			return float(memory_str[:-1]) * (1000 ** 2)
		elif memory_str.endswith('G'):
			return float(memory_str[:-1]) * (1000 ** 3)
		else:
			# Assume bytes
			return float(memory_str)

