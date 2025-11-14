"""Kubernetes actuator for executing plans."""

from __future__ import annotations

from typing import Dict, Optional
import uuid
import logging

from kubernetes import client, config
from kubernetes.client import V1Pod
from kubernetes.client.exceptions import ApiException

from dt.state import Plan, Job, PlacementDecision, DTState

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
    
    def __init__(self, namespace: str = "dt-fabric") -> None:
        """
        Initialize the actuator with Kubernetes client.
        
        Args:
            namespace: Kubernetes namespace for pod creation
        """
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
        self.namespace = namespace
    
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
                
                # Generate V1Pod object from placement decision
                pod = generate_pod_from_decision(
                    job_name=job.name,
                    decision=decision,
                    plan_id=plan_id,
                    namespace=self.namespace,
                    compute_cpu=compute_cpu,
                    compute_mem_gb=compute_mem_gb,
                )
                
                # Create pod in Kubernetes
                created_pod = self.core.create_namespaced_pod(
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
