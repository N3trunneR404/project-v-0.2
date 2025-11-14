"""Generate Kubernetes V1Pod objects from placement decisions."""

from __future__ import annotations

from typing import Optional
from kubernetes.client import V1Pod, V1ObjectMeta, V1PodSpec, V1Container, V1ResourceRequirements, V1EnvVar
from dt.state import PlacementDecision


def _get_image_for_format(exec_format: str) -> str:
    """Map execution format to container image."""
    if exec_format == "native":
        return "dt/worker-native:latest"
    elif exec_format.startswith("qemu-"):
        arch = exec_format.replace("qemu-", "")
        if arch == "arm64":
            return "dt/worker-qemu-arm64:latest"
        elif arch == "riscv64":
            return "dt/worker-qemu-riscv64:latest"
    elif exec_format == "wasm":
        return "dt/worker-wasm:latest"
    # Default fallback
    return "dt/worker-native:latest"


def _get_resource_requirements(compute_cpu: int = 1, compute_mem_gb: int = 1) -> V1ResourceRequirements:
    """Create resource requirements from compute specs."""
    return V1ResourceRequirements(
        requests={
            "cpu": f"{compute_cpu}",
            "memory": f"{compute_mem_gb}Gi",
        },
        limits={
            "cpu": f"{compute_cpu * 2}",  # Allow 2x for burst
            "memory": f"{compute_mem_gb * 2}Gi",
        },
    )


def generate_pod_from_decision(
    job_name: str,
    decision: PlacementDecision,
    plan_id: str,
    namespace: str = "dt-fabric",
    compute_cpu: int = 1,
    compute_mem_gb: int = 1,
) -> V1Pod:
    """
    Generate a V1Pod object from a placement decision.
    
    Args:
        job_name: Name of the job
        decision: Placement decision containing stage_id, node_name, exec_format
        plan_id: Plan ID for labeling
        namespace: Kubernetes namespace
        compute_cpu: CPU cores required (from JobStage.compute.cpu)
        compute_mem_gb: Memory in GB required (from JobStage.compute.mem_gb)
        
    Returns:
        V1Pod object ready for creation
    """
    pod_name = f"{job_name}-{decision.stage_id}-{plan_id[:8]}"
    
    # Determine container image based on exec format
    image = _get_image_for_format(decision.exec_format)
    
    # Create container with resource requests
    # Note: Resource values could come from job stage compute requirements
    # For now, use reasonable defaults
    container = V1Container(
        name="worker",
        image=image,
        image_pull_policy="IfNotPresent",
        resources=_get_resource_requirements(compute_cpu=compute_cpu, compute_mem_gb=compute_mem_gb),
        env=[
            V1EnvVar(name="STAGE_ID", value=decision.stage_id),
            V1EnvVar(name="EXEC_FORMAT", value=decision.exec_format),
            V1EnvVar(name="JOB_NAME", value=job_name),
            V1EnvVar(name="PLAN_ID", value=plan_id),
        ],
        command=["/worker.sh"],
    )
    
    # Create pod spec with node selector
    pod_spec = V1PodSpec(
        containers=[container],
        node_selector={
            "kubernetes.io/hostname": decision.node_name,
        },
        restart_policy="Never",
    )
    
    # Create pod metadata with labels
    pod_metadata = V1ObjectMeta(
        name=pod_name,
        namespace=namespace,
        labels={
            "dt.plan_id": plan_id,
            "dt.job_name": job_name,
            "dt.stage_id": decision.stage_id,
            "dt.exec_format": decision.exec_format,
            "dt.node_name": decision.node_name,
            "app": "dt-worker",
        },
    )
    
    # Create and return V1Pod
    pod = V1Pod(
        api_version="v1",
        kind="Pod",
        metadata=pod_metadata,
        spec=pod_spec,
    )
    
    return pod
