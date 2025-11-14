#!/bin/bash
set -euo pipefail

# Complete cluster setup for DT Fabric experiments
# Usage: ./scripts/setup-cluster.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Setting up DT Fabric Cluster ==="
echo ""

# Check if cluster exists
if ! kubectl cluster-info &>/dev/null; then
    echo "❌ Kubernetes cluster not accessible. Please ensure k3d cluster 'fabric-dt' is running."
    exit 1
fi

echo "✓ Cluster accessible"
echo ""

# Create namespace
echo "Creating namespace..."
kubectl create namespace dt-fabric --dry-run=client -o yaml | kubectl apply -f -
echo "✓ Namespace dt-fabric ready"
echo ""

# Install Node Feature Discovery
echo "Installing Node Feature Discovery (NFD)..."
if ! kubectl get daemonset -n kube-system node-feature-discovery-master &>/dev/null; then
    # Try to use helm if available, otherwise use kubectl apply
    if command -v helm &>/dev/null; then
        helm repo add nfd https://kubernetes-sigs.github.io/node-feature-discovery/charts 2>/dev/null || true
        helm repo update 2>/dev/null || true
        helm upgrade --install nfd nfd/node-feature-discovery \
            --namespace kube-system \
            --values "$PROJECT_ROOT/deploy/nfd-values.yaml" \
            --wait 2>/dev/null || echo "⚠ Helm installation failed, skipping NFD"
    else
        echo "⚠ Helm not found, installing NFD via kubectl..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/node-feature-discovery/v0.15.1/deployment/overlays/default/kustomization.yaml 2>/dev/null || \
        echo "⚠ NFD installation skipped (requires helm or manual installation)"
    fi
    echo "✓ NFD installation attempted"
else
    echo "✓ NFD already installed"
fi
echo ""

# Deploy netem DaemonSet for network emulation
echo "Deploying network emulation (netem)..."
kubectl apply -f "$PROJECT_ROOT/sim/network/netem-daemonset.yaml"
kubectl rollout status daemonset/netem-shaper -n kube-system --timeout=60s
echo "✓ Network emulation deployed"
echo ""

# Install Chaos Mesh (if not already installed)
echo "Checking Chaos Mesh installation..."
if ! kubectl get crd networkchaos.chaos-mesh.org &>/dev/null; then
    echo "Installing Chaos Mesh..."
    curl -sSL https://mirrors.chaos-mesh.org/latest/install.sh | bash
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=chaos-mesh -n chaos-mesh --timeout=300s || true
    echo "✓ Chaos Mesh installed"
else
    echo "✓ Chaos Mesh already installed"
fi
echo ""

# Deploy DT API
echo "Deploying DT API..."
kubectl apply -f "$PROJECT_ROOT/deploy/dt-api.yaml"
kubectl rollout status deployment/dt-api -n dt-fabric --timeout=120s
echo "✓ DT API deployed"
echo ""

# Wait for DT API to be ready
echo "Waiting for DT API to be ready..."
kubectl wait --for=condition=ready pod -l app=dt-api -n dt-fabric --timeout=120s
DT_API_POD=$(kubectl get pod -l app=dt-api -n dt-fabric -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n dt-fabric "$DT_API_POD" 8080:8080 &
PORT_FORWARD_PID=$!
sleep 5
echo "✓ DT API ready at http://127.0.0.1:8080 (port-forward PID: $PORT_FORWARD_PID)"
echo ""

# Show cluster status
echo "=== Cluster Status ==="
kubectl get nodes
echo ""
kubectl get pods -n dt-fabric
echo ""
kubectl get pods -n chaos-mesh
echo ""

echo "=== Setup Complete ==="
echo ""
echo "DT API: http://127.0.0.1:8080"
echo "Port-forward PID: $PORT_FORWARD_PID (kill with: kill $PORT_FORWARD_PID)"
echo ""
echo "Next steps:"
echo "  1. Run experiments: python experiments/run_suite.py"
echo "  2. Run chaos tests: kubectl apply -f chaos/scenarios/"

