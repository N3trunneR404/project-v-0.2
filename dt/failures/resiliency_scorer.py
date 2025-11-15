"""Resiliency scoring for clusters and nodes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from dt.state import DTState, Node

logger = logging.getLogger(__name__)


@dataclass
class ResiliencyMetrics:
	"""Metrics used for resiliency scoring."""
	failure_count: int = 0
	total_uptime_s: float = 0.0
	total_downtime_s: float = 0.0
	last_failure_time: Optional[float] = None
	avg_cpu_util: float = 0.0
	avg_mem_util: float = 0.0
	util_samples: int = 0


class ResiliencyScorer:
	"""
	Computes resiliency scores for clusters and nodes.
	
	Factors:
	- Historical failure rates
	- Current utilization
	- Node age/health metrics
	- Uptime percentage
	"""
	
	def __init__(self, state: DTState) -> None:
		"""
		Initialize resiliency scorer.
		
		Args:
			state: DTState to query for node information
		"""
		self.state = state
		self.node_metrics: Dict[str, ResiliencyMetrics] = {}
		self.cluster_metrics: Dict[str, ResiliencyMetrics] = {}
		
		logger.info("ResiliencyScorer initialized")
	
	def record_failure(self, node_name: str, cluster_name: Optional[str] = None) -> None:
		"""Record a failure event for a node."""
		current_time = time.time()
		
		# Update node metrics
		if node_name not in self.node_metrics:
			self.node_metrics[node_name] = ResiliencyMetrics()
		
		metrics = self.node_metrics[node_name]
		metrics.failure_count += 1
		metrics.last_failure_time = current_time
		
		# Update cluster metrics
		if cluster_name:
			if cluster_name not in self.cluster_metrics:
				self.cluster_metrics[cluster_name] = ResiliencyMetrics()
			self.cluster_metrics[cluster_name].failure_count += 1
			self.cluster_metrics[cluster_name].last_failure_time = current_time
		
		logger.debug(f"Recorded failure for node {node_name} (cluster: {cluster_name})")
	
	def record_uptime(self, node_name: str, uptime_s: float) -> None:
		"""Record uptime for a node."""
		if node_name not in self.node_metrics:
			self.node_metrics[node_name] = ResiliencyMetrics()
		
		self.node_metrics[node_name].total_uptime_s += uptime_s
	
	def record_downtime(self, node_name: str, downtime_s: float) -> None:
		"""Record downtime for a node."""
		if node_name not in self.node_metrics:
			self.node_metrics[node_name] = ResiliencyMetrics()
		
		self.node_metrics[node_name].total_downtime_s += downtime_s
	
	def update_utilization(self, node_name: str, cpu_util: float, mem_util: float) -> None:
		"""Update utilization metrics for a node."""
		if node_name not in self.node_metrics:
			self.node_metrics[node_name] = ResiliencyMetrics()
		
		metrics = self.node_metrics[node_name]
		
		# Update running average
		n = metrics.util_samples
		metrics.avg_cpu_util = (metrics.avg_cpu_util * n + cpu_util) / (n + 1)
		metrics.avg_mem_util = (metrics.avg_mem_util * n + mem_util) / (n + 1)
		metrics.util_samples += 1
	
	def compute_node_score(self, node_name: str) -> float:
		"""
		Compute resiliency score for a node (0.0 to 1.0).
		
		Higher score = more resilient.
		"""
		node = self.state.get_node(node_name)
		if not node:
			return 0.5  # Default score for unknown nodes
		
		metrics = self.node_metrics.get(node_name, ResiliencyMetrics())
		
		# Base score from availability
		total_time = metrics.total_uptime_s + metrics.total_downtime_s
		if total_time > 0:
			uptime_ratio = metrics.total_uptime_s / total_time
		else:
			uptime_ratio = 1.0  # No history, assume good
		
		# Penalty for failures
		failure_penalty = min(0.3, metrics.failure_count * 0.05)
		
		# Penalty for high utilization (stress)
		util_penalty = 0.0
		if metrics.avg_cpu_util > 80.0:
			util_penalty += 0.1
		if metrics.avg_mem_util > 80.0:
			util_penalty += 0.1
		
		# Penalty for recent failures
		recent_failure_penalty = 0.0
		if metrics.last_failure_time:
			time_since_failure = time.time() - metrics.last_failure_time
			if time_since_failure < 3600:  # Within last hour
				recent_failure_penalty = 0.2
			elif time_since_failure < 86400:  # Within last day
				recent_failure_penalty = 0.1
		
		# Compute final score
		score = uptime_ratio - failure_penalty - util_penalty - recent_failure_penalty
		return max(0.0, min(1.0, score))
	
	def compute_cluster_score(self, cluster_name: str) -> float:
		"""
		Compute resiliency score for a cluster (0.0 to 1.0).
		
		Higher score = more resilient.
		"""
		cluster_info = self.state.clusters.get(cluster_name)
		if not cluster_info:
			return 0.5  # Default score
		
		# Aggregate node scores
		node_scores = []
		for node_name in cluster_info.nodes:
			score = self.compute_node_score(node_name)
			node_scores.append(score)
		
		if not node_scores:
			return 0.5
		
		# Cluster score is average of node scores, weighted by cluster metrics
		avg_node_score = sum(node_scores) / len(node_scores)
		
		# Apply cluster-level penalties
		cluster_metrics = self.cluster_metrics.get(cluster_name, ResiliencyMetrics())
		failure_penalty = min(0.2, cluster_metrics.failure_count * 0.03)
		
		score = avg_node_score - failure_penalty
		return max(0.0, min(1.0, score))
	
	def update_cluster_score(self, cluster_name: str, state: DTState) -> None:
		"""Update resiliency score in cluster info."""
		score = self.compute_cluster_score(cluster_name)
		cluster_info = state.clusters.get(cluster_name)
		if cluster_info:
			cluster_info.resiliency_score = score
			logger.debug(f"Updated resiliency score for cluster {cluster_name}: {score:.3f}")

