from __future__ import annotations

import json
from typing import Any, Dict, List
from flask import Flask, request, jsonify

# Minimal scheduler extender-like scoring service
# Note: Modern K8s prefers Scheduler Framework plugins (Go). This HTTP scorer
# is provided for experimentation only.

app = Flask(__name__)


@app.post("/filter")
def filter_nodes():
	body = request.get_json(force=True)
	nodes = body.get("nodes", {}).get("items", [])
	# Pass-through filter, keep all nodes
	return jsonify({"nodes": nodes, "failedNodes": {}, "error": ""})


@app.post("/prioritize")
def prioritize_nodes():
	body = request.get_json(force=True)
	nodes = body.get("nodes", {}).get("items", [])
	# Assign a flat score; a real implementation would query DT scores
	priorities = [{"name": n["metadata"]["name"], "score": 50} for n in nodes]
	return jsonify({"priorities": priorities})


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8090)





