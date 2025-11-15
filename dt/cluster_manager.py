"""Multi-cluster manager for Digital Twin."""

from __future__ import annotations

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path

from kubernetes import client, config
from kubernetes.client import ApiClient
from kubernetes.client.exceptions import ApiException

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Metadata for a Kubernetes cluster."""
    name: str
    cluster_type: str  # datacenter, mining, lab, gaming, pan, edge
    kubeconfig_path: Optional[str] = None
    api_client: Optional[ApiClient] = None
    core_api: Optional[client.CoreV1Api] = None
    metrics_api: Optional[client.CustomObjectsApi] = None
    resiliency_score: float = 0.8  # Initial score, updated by resiliency_scorer
    nodes: list[str] = field(default_factory=list)


class ClusterManager:
    """Manages connections to multiple Kubernetes clusters."""
    
    def __init__(self, latency_matrix_path: Optional[str] = None):
        """
        Initialize cluster manager.
        
        Args:
            latency_matrix_path: Path to latency-matrix.yaml file
        """
        self.clusters: Dict[str, ClusterInfo] = {}
        self.latency_matrix: Dict[Tuple[str, str], float] = {}
        self.default_latency_ms: float = 100.0
        self.intra_cluster_latency_ms: float = 0.5
        
        if latency_matrix_path:
            self._load_latency_matrix(latency_matrix_path)
        
        # Auto-discover clusters from k3d
        self._discover_clusters()
    
    def _load_latency_matrix(self, path: str) -> None:
        """Load inter-cluster latency matrix from YAML file."""
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            latencies = data.get('latencies', [])
            for entry in latencies:
                from_cluster = entry['from']
                to_cluster = entry['to']
                latency = float(entry['latency_ms'])
                
                # Store both directions (symmetric)
                self.latency_matrix[(from_cluster, to_cluster)] = latency
                self.latency_matrix[(to_cluster, from_cluster)] = latency
            
            self.default_latency_ms = float(data.get('default_latency_ms', 100.0))
            self.intra_cluster_latency_ms = float(data.get('intra_cluster_latency_ms', 0.5))
            
            logger.info(f"Loaded latency matrix: {len(latencies)} entries, default={self.default_latency_ms}ms")
        except Exception as e:
            logger.warning(f"Failed to load latency matrix from {path}: {e}")
    
    def _discover_clusters(self) -> None:
        """Discover k3d clusters and initialize connections."""
        # Try to discover clusters via k3d
        # k3d stores kubeconfigs in ~/.config/k3d/<cluster-name>/kubeconfig.yaml
        k3d_config_dir = Path.home() / ".config" / "k3d"
        
        if not k3d_config_dir.exists():
            logger.warning("k3d config directory not found, clusters must be registered manually")
            return
        
        # Expected cluster names
        expected_clusters = [
            "dc-core", "prosumer-mining", "campus-lab", "gamer-pc",
            "phone-pan-1", "phone-pan-2", "edge-microdc"
        ]
        
        cluster_type_map = {
            "dc-core": "datacenter",
            "prosumer-mining": "mining",
            "campus-lab": "lab",
            "gamer-pc": "gaming",
            "phone-pan-1": "pan",
            "phone-pan-2": "pan",
            "edge-microdc": "edge",
        }
        
        for cluster_name in expected_clusters:
            kubeconfig_path = k3d_config_dir / cluster_name / "kubeconfig.yaml"
            
            if kubeconfig_path.exists():
                self.register_cluster(
                    cluster_name,
                    cluster_type_map.get(cluster_name, "unknown"),
                    str(kubeconfig_path)
                )
            else:
                logger.debug(f"Cluster '{cluster_name}' not found at {kubeconfig_path}")
    
    def register_cluster(
        self,
        name: str,
        cluster_type: str,
        kubeconfig_path: Optional[str] = None
    ) -> ClusterInfo:
        """
        Register a cluster and initialize Kubernetes client.
        
        Args:
            name: Cluster name
            cluster_type: Type of cluster (datacenter, mining, lab, etc.)
            kubeconfig_path: Path to kubeconfig file (optional, uses default if None)
            
        Returns:
            ClusterInfo for the registered cluster
        """
        if name in self.clusters:
            logger.info(f"Cluster '{name}' already registered, updating connection")
            cluster_info = self.clusters[name]
        else:
            cluster_info = ClusterInfo(name=name, cluster_type=cluster_type)
            self.clusters[name] = cluster_info
        
        # Initialize Kubernetes client
        try:
            if kubeconfig_path:
                # Use specific kubeconfig
                os.environ['KUBECONFIG'] = kubeconfig_path
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                # Try to use default kubeconfig
                try:
                    config.load_kube_config()
                except config.ConfigException:
                    # Try incluster config as fallback
                    config.load_incluster_config()
            
            api_client = ApiClient()
            cluster_info.api_client = api_client
            cluster_info.core_api = client.CoreV1Api(api_client)
            cluster_info.metrics_api = client.CustomObjectsApi(api_client)
            cluster_info.kubeconfig_path = kubeconfig_path
            
            # List nodes to verify connection
            nodes = cluster_info.core_api.list_node()
            cluster_info.nodes = [node.metadata.name for node in nodes.items]
            
            logger.info(f"Registered cluster '{name}' ({cluster_type}): {len(cluster_info.nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Failed to initialize cluster '{name}': {e}")
            raise
        
        return cluster_info
    
    def get_cluster(self, cluster_name: str) -> Optional[ClusterInfo]:
        """Get cluster info by name."""
        return self.clusters.get(cluster_name)
    
    def get_cluster_for_node(self, node_name: str) -> Optional[str]:
        """
        Determine which cluster a node belongs to.
        
        Args:
            node_name: Kubernetes node name
            
        Returns:
            Cluster name or None if not found
        """
        for cluster_name, cluster_info in self.clusters.items():
            if node_name in cluster_info.nodes:
                return cluster_name
        
        # Try to infer from node name pattern
        # e.g., "dc-core-worker-0" -> "dc-core"
        for cluster_name in self.clusters.keys():
            if node_name.startswith(cluster_name):
                return cluster_name
        
        return None
    
    def get_latency_between(
        self,
        cluster_a: str,
        cluster_b: str,
        node_a: Optional[str] = None,
        node_b: Optional[str] = None
    ) -> float:
        """
        Get latency between two clusters or nodes.
        
        Args:
            cluster_a: Source cluster name
            cluster_b: Destination cluster name
            node_a: Optional source node name (for future intra-cluster routing)
            node_b: Optional destination node name
            
        Returns:
            Latency in milliseconds
        """
        # Same cluster: intra-cluster latency
        if cluster_a == cluster_b:
            return self.intra_cluster_latency_ms
        
        # Look up in latency matrix
        key = (cluster_a, cluster_b)
        if key in self.latency_matrix:
            return self.latency_matrix[key]
        
        # Fallback to default
        logger.debug(f"Latency not found for {cluster_a} -> {cluster_b}, using default {self.default_latency_ms}ms")
        return self.default_latency_ms
    
    def get_api_client(self, cluster_name: str) -> Optional[ApiClient]:
        """Get Kubernetes API client for a cluster."""
        cluster = self.get_cluster(cluster_name)
        return cluster.api_client if cluster else None
    
    def get_core_api(self, cluster_name: str) -> Optional[client.CoreV1Api]:
        """Get CoreV1Api client for a cluster."""
        cluster = self.get_cluster(cluster_name)
        return cluster.core_api if cluster else None
    
    def list_clusters(self) -> list[str]:
        """List all registered cluster names."""
        return list(self.clusters.keys())
    
    def update_resiliency_score(self, cluster_name: str, score: float) -> None:
        """Update resiliency score for a cluster."""
        cluster = self.get_cluster(cluster_name)
        if cluster:
            cluster.resiliency_score = max(0.0, min(1.0, score))
            logger.debug(f"Updated resiliency score for '{cluster_name}': {score:.3f}")

