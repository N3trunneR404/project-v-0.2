"""Failure event generator (time-based and event-driven)."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from dt.state import DTState
from dt.failures.resiliency_scorer import ResiliencyScorer

logger = logging.getLogger(__name__)


class FailureType(Enum):
	"""Types of failure events."""
	NODE_DOWN = "node_down"
	THERMAL_THROTTLE = "thermal_throttle"
	NETWORK_DEGRADATION = "network_degradation"
	ZOMBIE_PROCESS = "zombie_process"
	SYSTEM_CRASH = "system_crash"


@dataclass
class FailureEvent:
	"""A failure event to inject."""
	event_type: FailureType
	target_cluster: Optional[str] = None
	target_node: Optional[str] = None
	severity: float = 1.0  # 0.0 to 1.0
	timestamp: float = 0.0
	
	def __post_init__(self) -> None:
		if self.timestamp == 0.0:
			self.timestamp = time.time()


class FailureEventGenerator:
	"""
	Generates failure events using time-based and event-driven triggers.
	
	Time-based: Random events at intervals, weighted by resiliency score
	Event-driven: Triggered by telemetry thresholds (CPU > 80%, etc.)
	"""
	
	def __init__(
		self,
		state: DTState,
		resiliency_scorer: ResiliencyScorer,
		time_interval_s: float = 60.0,
		enable_time_based: bool = True,
		enable_event_driven: bool = True,
	) -> None:
		"""
		Initialize failure event generator.
		
		Args:
			state: DTState to query for nodes/clusters
			resiliency_scorer: ResiliencyScorer for computing scores
			time_interval_s: Interval for time-based failures (default 60s)
			enable_time_based: Enable time-based failures
			enable_event_driven: Enable event-driven failures
		"""
		self.state = state
		self.resiliency_scorer = resiliency_scorer
		self.time_interval_s = time_interval_s
		self.enable_time_based = enable_time_based
		self.enable_event_driven = enable_event_driven
		
		self._running = False
		self._thread: Optional[threading.Thread] = None
		self._stop_event = threading.Event()
		
		# Event callbacks
		self.on_failure: Optional[Callable[[FailureEvent], None]] = None
		
		# Thresholds for event-driven failures
		self.cpu_threshold = 80.0  # CPU utilization %
		self.mem_threshold = 80.0  # Memory utilization %
		self.queue_length_threshold = 10  # Pod queue length
		
		# Track last failure times to avoid spam
		self._last_failure_time: Dict[str, float] = {}
		self._min_failure_interval_s = 30.0
		
		logger.info(
			f"FailureEventGenerator initialized: "
			f"time_based={enable_time_based}, event_driven={enable_event_driven}"
		)
	
	def start(self) -> None:
		"""Start generating failure events."""
		if self._running:
			logger.warning("FailureEventGenerator already running")
			return
		
		self._running = True
		self._stop_event.clear()
		
		if self.enable_time_based:
			self._thread = threading.Thread(
				target=self._time_based_loop,
				name="failure-generator",
				daemon=True
			)
			self._thread.start()
			logger.info("FailureEventGenerator started (time-based)")
	
	def stop(self) -> None:
		"""Stop generating failure events."""
		if not self._running:
			return
		
		self._running = False
		self._stop_event.set()
		
		if self._thread:
			self._thread.join(timeout=5.0)
		
		logger.info("FailureEventGenerator stopped")
	
	def check_event_driven(self) -> List[FailureEvent]:
		"""
		Check for event-driven failure conditions.
		
		Returns:
			List of failure events to inject
		"""
		if not self.enable_event_driven:
			return []
		
		events: List[FailureEvent] = []
		
		# Check node utilization
		for node in self.state.list_nodes():
			# Check CPU threshold
			if node.tel.cpu_util > self.cpu_threshold:
				event = self._create_failure_event(
					FailureType.THERMAL_THROTTLE,
					target_node=node.name,
					severity=min(1.0, (node.tel.cpu_util - self.cpu_threshold) / 20.0)
				)
				if event:
					events.append(event)
			
			# Check memory threshold
			if node.tel.mem_util > self.mem_threshold:
				event = self._create_failure_event(
					FailureType.SYSTEM_CRASH,
					target_node=node.name,
					severity=min(1.0, (node.tel.mem_util - self.mem_threshold) / 20.0)
				)
				if event:
					events.append(event)
		
		return events
	
	def _time_based_loop(self) -> None:
		"""Main loop for time-based failure generation."""
		while not self._stop_event.is_set():
			try:
				# Generate random failures weighted by resiliency
				events = self._generate_time_based_failures()
				for event in events:
					if self.on_failure:
						try:
							self.on_failure(event)
						except Exception as e:
							logger.error(f"Error in failure callback: {e}")
			
			except Exception as e:
				logger.error(f"Error in time-based failure loop: {e}")
			
			# Wait for next interval
			self._stop_event.wait(self.time_interval_s)
	
	def _generate_time_based_failures(self) -> List[FailureEvent]:
		"""Generate time-based failures weighted by resiliency scores."""
		events: List[FailureEvent] = []
		
		# Iterate through clusters
		for cluster_name, cluster_info in self.state.clusters.items():
			# Lower resiliency score = higher chance of failure
			resiliency = cluster_info.resiliency_score
			failure_probability = (1.0 - resiliency) * 0.3  # Max 30% chance
			
			if random.random() < failure_probability:
				# Select random node in cluster
				if cluster_info.nodes:
					target_node = random.choice(cluster_info.nodes)
					
					# Select failure type based on cluster type
					event_type = self._select_failure_type(cluster_info.cluster_type)
					
					event = self._create_failure_event(
						event_type,
						target_cluster=cluster_name,
						target_node=target_node,
						severity=1.0 - resiliency
					)
					if event:
						events.append(event)
		
		return events
	
	def _select_failure_type(self, cluster_type: str) -> FailureType:
		"""Select failure type based on cluster characteristics."""
		# Weighted selection based on cluster type
		if cluster_type == "gaming":
			# Gaming PCs: more likely to have thermal issues
			return random.choice([
				FailureType.THERMAL_THROTTLE,
				FailureType.NODE_DOWN,
				FailureType.SYSTEM_CRASH,
			])
		elif cluster_type == "pan":
			# PANs: more likely network issues
			return random.choice([
				FailureType.NETWORK_DEGRADATION,
				FailureType.NODE_DOWN,
			])
		elif cluster_type == "mining":
			# Mining rigs: thermal and system crashes
			return random.choice([
				FailureType.THERMAL_THROTTLE,
				FailureType.SYSTEM_CRASH,
			])
		else:
			# Default: mix of all types
			return random.choice(list(FailureType))
	
	def _create_failure_event(
		self,
		event_type: FailureType,
		target_cluster: Optional[str] = None,
		target_node: Optional[str] = None,
		severity: float = 1.0
	) -> Optional[FailureEvent]:
		"""Create a failure event if not too recent."""
		# Check if we recently generated a failure for this target
		target_key = target_node or target_cluster or "unknown"
		last_time = self._last_failure_time.get(target_key, 0.0)
		if time.time() - last_time < self._min_failure_interval_s:
			return None
		
		# Create event
		event = FailureEvent(
			event_type=event_type,
			target_cluster=target_cluster,
			target_node=target_node,
			severity=severity,
		)
		
		# Update last failure time
		self._last_failure_time[target_key] = time.time()
		
		return event

