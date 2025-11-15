"""Kubernetes actuator for executing plans."""

from __future__ import annotations

from typing import Dict, Optional
import uuid
import logging

from kubernetes import client, config
from kubernetes.client import V1Pod
from kubernetes.client.exceptions import ApiException

from dt.state import Plan, Job, PlacementDecision, DTState
from dt.cluster_manager import ClusterManager
from dt.failures.event_generator import FailureEvent, FailureType

logger = logging.getLogger(__name__)

try:
    from k8s_executor.pod_gen import generate_pod_from_decision
except ImportError:
    # k8s_executor not available - use stub
    logger.warning("k8s_executor.pod_gen not available, using stub")
    def generate_pod_from_decision(
        job_name: str,
        decision: PlacementDecision,
        plan_id: str,
        namespace: str = "dt-fabric",
    ) -> V1Pod:
        """Stub implementation when k8s_executor is not available."""
        from kubernetes.client import V1Pod, V1ObjectMeta, V1PodSpec, V1Container
        return V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=V1ObjectMeta(
                name=f"{job_name}-{decision.stage_id}-{plan_id[:8]}",
                namespace=namespace,
                labels={
                    "dt.plan_id": plan_id,
                    "dt.job_name": job_name,
                    "dt.stage_id": decision.stage_id,
                    "dt.exec_format": decision.exec_format,
                },
            ),
            spec=V1PodSpec(
                containers=[V1Container(name="worker", image="dt/worker-native:latest")],
            ),
        )


class Actuator:
    """Kubernetes actuator for executing Digital Twin plans."""
    
    def __init__(
        self,
        namespace: str = "dt-fabric",
        cluster_manager: Optional[ClusterManager] = None
    ) -> None:
        """
        Initialize the actuator with Kubernetes client.
        
        Args:
            namespace: Kubernetes namespace for pod creation
            cluster_manager: Optional ClusterManager for multi-cluster support
        """
        self.cluster_manager = cluster_manager
        self.namespace = namespace
        
        # Initialize default client (for single-cluster mode)
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except Exception:
            try:
                config.load_kube_config()
                logger.info("Loaded kubeconfig")
            except Exception as e:
                logger.warning(f"Could not load Kubernetes config: {e}")
        
        self.core = client.CoreV1Api()
    
    def _node_exists(self, node_name: str) -> bool:
        """
        Check if a node exists in the cluster.
        
        Args:
            node_name: Name of the node to check
            
        Returns:
            True if node exists, False otherwise
        """
        try:
            self.core.read_node(node_name)
            return True
        except ApiException as e:
            if e.status == 404:
                return False
            # Re-raise other API exceptions
            raise
        except Exception as e:
            logger.error(f"Error checking node existence: {e}")
            return False
    
    def cordon_node(self, node_name: str) -> None:
        """
        Cordon a node (mark as unschedulable).
        
        Args:
            node_name: Name of the node to cordon
            
        Raises:
            ValueError: If node does not exist
            ApiException: If Kubernetes API call fails
        """
        if not self._node_exists(node_name):
            raise ValueError(f"Node '{node_name}' does not exist")
        
        try:
            body = {"spec": {"unschedulable": True}}
            self.core.patch_node(node_name, body)
            logger.info(f"Cordoned node: {node_name}")
        except ApiException as e:
            logger.error(f"Failed to cordon node {node_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error cordoning node {node_name}: {e}")
            raise
    
    def uncordon_node(self, node_name: str) -> None:
        """
        Uncordon a node (mark as schedulable).
        
        Args:
            node_name: Name of the node to uncordon
            
        Raises:
            ValueError: If node does not exist
            ApiException: If Kubernetes API call fails
        """
        if not self._node_exists(node_name):
            raise ValueError(f"Node '{node_name}' does not exist")
        
        try:
            body = {"spec": {"unschedulable": False}}
            self.core.patch_node(node_name, body)
            logger.info(f"Uncordoned node: {node_name}")
        except ApiException as e:
            logger.error(f"Failed to uncordon node {node_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uncordoning node {node_name}: {e}")
            raise
    
    def submit_plan(self, job: Job, placements: Dict[str, PlacementDecision], plan_id: Optional[str] = None) -> Plan:
        """
        Submit a plan by creating Kubernetes pods for each placement.
        
        Args:
            job: Job specification
            placements: Dictionary mapping stage_id to PlacementDecision
            plan_id: Optional plan ID (will be generated if not provided)
            
        Returns:
            Plan object with plan_id
            
        Raises:
            ApiException: If pod creation fails
        """
        if plan_id is None:
            plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        
        created_pods = []
        errors = []
        
        for stage_id, decision in placements.items():
            try:
                # Find the job stage to get compute requirements
                stage = next((s for s in job.stages if s.id == stage_id), None)
                compute_cpu = stage.compute.cpu if stage else 1
                compute_mem_gb = stage.compute.mem_gb if stage else 1
                
                # Determine target cluster
                target_cluster = None
                core_api = self.core
                
                if self.cluster_manager:
                    # Get cluster for the target node
                    target_cluster = self.cluster_manager.get_cluster_for_node(decision.node_name)
                    if target_cluster:
                        # Get cluster-specific API client
                        cluster_core_api = self.cluster_manager.get_core_api(target_cluster)
                        if cluster_core_api:
                            core_api = cluster_core_api
                            logger.info(f"Using cluster '{target_cluster}' for node '{decision.node_name}'")
                
                # Generate V1Pod object from placement decision
                # Use resource scaling (default 1:100)
                from dt.scaling import DEFAULT_SCALER
                pod = generate_pod_from_decision(
                    job_name=job.name,
                    decision=decision,
                    plan_id=plan_id,
                    namespace=self.namespace,
                    compute_cpu=compute_cpu,
                    compute_mem_gb=compute_mem_gb,
                    duration_ms=stage.compute.duration_ms if stage else None,
                    resource_scale=DEFAULT_SCALER.cpu_scale,
                )
                
                # Create pod in Kubernetes (in correct cluster)
                created_pod = core_api.create_namespaced_pod(
                    namespace=self.namespace,
                    body=pod,
                )
                created_pods.append(created_pod.metadata.name)
                logger.info(
                    f"Created pod {created_pod.metadata.name} for job {job.name}, "
                    f"stage {stage_id} on node {decision.node_name}"
                )
            except ApiException as e:
                error_msg = (
                    f"Failed to create pod for stage {stage_id}: "
                    f"status={e.status}, reason={e.reason}"
                )
                logger.error(error_msg)
                errors.append(error_msg)
                # Continue with other pods even if one fails
            except Exception as e:
                error_msg = f"Unexpected error creating pod for stage {stage_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if errors and not created_pods:
            # All pods failed - raise an exception
            raise RuntimeError(f"Failed to create any pods: {'; '.join(errors)}")
        
        if errors:
            # Some pods failed - log warning but continue
            logger.warning(f"Some pods failed to create: {'; '.join(errors)}")
        
        return Plan(
            plan_id=plan_id,
            job_name=job.name,
            placements=placements,
            predicted_latency_ms=0.0,
            predicted_energy_kwh=0.0,
            risk_score=0.0,
        )
    
    def inject_failure(self, failure_event: FailureEvent) -> None:
        """
        Inject a failure event into the cluster.
        
        Args:
            failure_event: Failure event to inject
        """
        try:
            if failure_event.event_type == FailureType.NODE_DOWN:
                if failure_event.target_node:
                    self.cordon_node(failure_event.target_node)
                    logger.info(f"Injected node_down failure for {failure_event.target_node}")
            
            elif failure_event.event_type == FailureType.THERMAL_THROTTLE:
                # Simulate thermal throttling by reducing allocatable CPU
                # This would require patching the node's allocatable resources
                # For now, we just cordon the node as a proxy
                if failure_event.target_node:
                    logger.warning(f"Thermal throttle for {failure_event.target_node} (simulated as cordon)")
                    self.cordon_node(failure_event.target_node)
            
            elif failure_event.event_type == FailureType.NETWORK_DEGRADATION:
                # Network degradation would be handled by netem or Chaos Mesh
                logger.info(f"Network degradation for {failure_event.target_cluster} (requires netem/Chaos Mesh)")
            
            elif failure_event.event_type == FailureType.SYSTEM_CRASH:
                # System crash: cordon node and optionally kill pods
                if failure_event.target_node:
                    self.cordon_node(failure_event.target_node)
                    # Optionally kill running pods on this node
                    logger.info(f"Injected system_crash failure for {failure_event.target_node}")
            
            else:
                logger.warning(f"Unknown failure type: {failure_event.event_type}")
        
        except Exception as e:
            logger.error(f"Failed to inject failure {failure_event.event_type}: {e}")
