from __future__ import annotations

from typing import Dict
from dt.state import PlacementDecision


def generate_pod_from_decision(job_name: str, dec: PlacementDecision) -> Dict:
	image = _image_for(dec.exec_format)
	pod = {
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": f"{job_name}-{dec.stage_id}-{dec.node_name}",
			"labels": {"job": job_name, "stage": dec.stage_id},
		},
		"spec": {
			"nodeName": dec.node_name,
			"restartPolicy": "Never",
			"containers": [
				{
					"name": "worker",
					"image": image,
					"imagePullPolicy": "IfNotPresent",
					"env": [
						{"name": "STAGE_ID", "value": dec.stage_id},
						{"name": "EXEC_FORMAT", "value": dec.exec_format},
					],
					"resources": {"requests": {"cpu": "1", "memory": "512Mi"}},
					"command": ["/worker.sh"],
				}
			],
		},
	}
	return pod


def _image_for(exec_format: str) -> str:
	if exec_format.startswith("qemu-"):
		arch = exec_format.split("-", 1)[1]
		return f"worker-qemu-{arch}:latest"
	if exec_format == "wasm":
		return "worker-wasm:latest"
	return "worker-native:latest"





