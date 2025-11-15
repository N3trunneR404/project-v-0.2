#!/bin/bash
set -euo pipefail

# Multi-cluster setup for Digital Twin Fabric
# Creates 6-7 k3d clusters representing different device pools
# Usage: ./deploy/multi-cluster-setup.sh [--clean] [--skip-metrics]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CLEAN=false
SKIP_METRICS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Digital Twin Multi-Cluster Setup ==="
echo ""

# Cluster definitions: name, type, servers, agents, characteristics
declare -A CLUSTERS=(
    ["dc-core"]="datacenter:1:3:high_cpu:gpu:ssd"
    ["prosumer-mining"]="mining:1:2:high_cpu:gpu:high_power"
    ["campus-lab"]="lab:1:2:standard:no_gpu:standard"
    ["gamer-pc"]="gaming:1:2:high_cpu:gpu:ssd:unreliable"
    ["phone-pan-1"]="pan:1:3:low_cpu:no_gpu:low_power"
    ["phone-pan-2"]="pan:1:3:low_cpu:no_gpu:low_power"
    ["edge-microdc"]="edge:1:2:standard:no_gpu:standard"
)

if [ "$CLEAN" = true ]; then
    echo "Cleaning up existing clusters..."
    for cluster_name in "${!CLUSTERS[@]}"; do
        if k3d cluster list | grep -q "$cluster_name"; then
            echo "  Deleting cluster: $cluster_name"
            k3d cluster delete "$cluster_name" || true
        fi
    done
    echo "✓ Cleanup complete"
    echo ""
fi

# Create clusters
echo "Creating clusters..."
for cluster_name in "${!CLUSTERS[@]}"; do
    IFS=':' read -r cluster_type servers agents rest <<< "${CLUSTERS[$cluster_name]}"
    
    if k3d cluster list | grep -q "$cluster_name"; then
        echo "  ✓ Cluster '$cluster_name' already exists (skipping)"
        continue
    fi
    
    echo "  Creating cluster: $cluster_name (type: $cluster_type, $servers server, $agents agents)"
    
    # Create cluster config
    CLUSTER_CONFIG="$PROJECT_ROOT/deploy/cluster-configs/${cluster_name}.yaml"
    mkdir -p "$(dirname "$CLUSTER_CONFIG")"
    
    cat > "$CLUSTER_CONFIG" <<EOF
apiVersion: k3d.io/v1alpha5
kind: Simple
metadata:
  name: ${cluster_name}
servers: ${servers}
agents: ${agents}
options:
  k3s:
    extraArgs:
      - arg: "--kube-apiserver-arg=enable-admission-plugins=MutatingAdmissionWebhook,ValidatingAdmissionWebhook"
        nodeFilters:
          - server:*
  kubeconfig:
    updateDefaultKubeconfig: false
    switchCurrentContext: false
labels:
  - key: "dt.cluster.type"
    value: "${cluster_type}"
  - key: "dt.cluster.name"
    value: "${cluster_name}"
EOF
    
    # Create cluster
    k3d cluster create "$cluster_name" --config "$CLUSTER_CONFIG"
    
    # Set kubeconfig context
    k3d kubeconfig merge "$cluster_name" --kubeconfig-merge-default || true
    
    echo "    ✓ Cluster '$cluster_name' created"
done

echo ""
echo "=== Installing Metrics Server ==="

# Install metrics-server in each cluster
for cluster_name in "${!CLUSTERS[@]}"; do
    if [ "$SKIP_METRICS" = true ]; then
        echo "  Skipping metrics-server installation (--skip-metrics)"
        break
    fi
    
    echo "  Installing metrics-server in '$cluster_name'..."
    
    # Switch context
    export KUBECONFIG="$(k3d kubeconfig write "$cluster_name")"
    
    # Install metrics-server
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml || true
    
    # Patch to allow insecure TLS (for k3d)
    kubectl patch deployment metrics-server -n kube-system --type=json \
        -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]' || true
    
    # Wait for metrics-server to be ready
    kubectl wait --for=condition=available deployment/metrics-server -n kube-system --timeout=120s || true
    
    echo "    ✓ Metrics-server installed in '$cluster_name'"
done

# Reset KUBECONFIG
unset KUBECONFIG

echo ""
echo "=== Cluster Status ==="
for cluster_name in "${!CLUSTERS[@]}"; do
    echo ""
    echo "Cluster: $cluster_name"
    export KUBECONFIG="$(k3d kubeconfig write "$cluster_name")"
    kubectl get nodes || echo "  (unable to connect)"
    unset KUBECONFIG
done

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Clusters created:"
for cluster_name in "${!CLUSTERS[@]}"; do
    echo "  - $cluster_name"
done
echo ""
echo "To use a specific cluster:"
echo "  export KUBECONFIG=\$(k3d kubeconfig write <cluster-name>)"
echo ""
echo "Next steps:"
echo "  1. Configure latency matrix: deploy/latency-matrix.yaml"
echo "  2. Initialize cluster manager in DT API"
echo "  3. Run experiments with multi-cluster support"

