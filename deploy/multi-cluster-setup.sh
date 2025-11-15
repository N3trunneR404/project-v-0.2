#!/bin/bash
set -euo pipefail

# Multi-cluster setup for Digital Twin Fabric
# Creates multiple k3d clusters representing distinct device pools with
# per-cluster network isolation and deterministic virtual IDs.
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

# Cluster definitions: name|type|servers|agents|cluster_id|network_cidr|pod_cidr|service_cidr|accelerators
CLUSTER_SPECS=(
    "dc-core|datacenter|1|5|10|172.30.10.0/24|10.10.0.0/16|10.110.0.0/16|gpu:nvidia-a100"
    "prosumer-mining|mining|1|4|11|172.30.11.0/24|10.11.0.0/16|10.111.0.0/16|gpu:nvidia-rtx"
    "campus-lab|lab|1|3|12|172.30.12.0/24|10.12.0.0/16|10.112.0.0/16|gpu:nvidia-t4"
    "edge-microdc|edge|1|6|13|172.30.13.0/24|10.13.0.0/16|10.113.0.0/16|npu:habana-gaudi"
    "phone-pan-1|pan|1|4|14|172.30.14.0/24|10.14.0.0/16|10.114.0.0/16|npu:qualcomm-hvx"
    "phone-pan-2|pan|1|4|15|172.30.15.0/24|10.15.0.0/16|10.115.0.0/16|gpu:adreno"
    "gamer-pc|gaming|1|4|16|172.30.16.0/24|10.16.0.0/16|10.116.0.0/16|gpu:nvidia-rtx4090"
)

derive_ip() {
    local cidr="$1"
    local index="$2"
    python3 - <<PY
import ipaddress
cidr = ipaddress.ip_network("$cidr", strict=False)
index = int($index)
host_index = max(10, index + 10)
if host_index >= cidr.num_addresses - 1:
    host_index = (index % (cidr.num_addresses - 2)) + 1
addr = cidr.network_address + host_index
if addr == cidr.broadcast_address:
    addr -= 1
print(str(addr))
PY
}

annotate_nodes() {
    local cluster_name="$1"
    local cluster_type="$2"
    local cluster_id="$3"
    local pod_cidr="$4"
    local service_cidr="$5"
    local network_cidr="$6"
    local accelerator_hint="$7"

    local nodes node_id=0
    nodes=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
    for node in $nodes; do
        node_id=$((node_id + 1))
        local pod_ip
        pod_ip=$(derive_ip "$pod_cidr" "$node_id")
        local node_ip
        node_ip=$(derive_ip "$network_cidr" "$node_id")
        local mac
        mac=$(printf '02:fd:%02x:%02x:%02x:%02x' "$cluster_id" "$(((node_id >> 8) & 0xff))" "$((node_id & 0xff))" "$(((node_id * 37) & 0xff))")

        kubectl label node "$node" \
            dt.cluster.name="$cluster_name" \
            dt.cluster.type="$cluster_type" \
            dt.virtual.cluster_id="$cluster_id" \
            dt.virtual.node_id="$node_id" \
            dt.node.poolHint="${accelerator_hint}" \
            --overwrite >/dev/null 2>&1 || true

        kubectl annotate node "$node" \
            dt.virtual.pod_cidr="$pod_cidr" \
            dt.virtual.service_cidr="$service_cidr" \
            dt.virtual.network_cidr="$network_cidr" \
            dt.virtual.pod_ip="$pod_ip" \
            dt.virtual.node_ip="$node_ip" \
            dt.virtual.mac="$mac" \
            --overwrite >/dev/null 2>&1 || true
    done
}

create_cluster() {
    local name="$1"
    local type="$2"
    local servers="$3"
    local agents="$4"
    local cluster_id="$5"
    local network_cidr="$6"
    local pod_cidr="$7"
    local service_cidr="$8"
    local accelerator_hint="$9"

    if k3d cluster list | grep -q "^${name}[[:space:]]"; then
        echo "  ✓ Cluster '$name' already exists (skipping)"
        return
    fi

    echo "  Creating cluster: $name (type: $type, $servers server, $agents agents)"

    local cluster_config="$PROJECT_ROOT/deploy/cluster-configs/${name}.yaml"
    mkdir -p "$(dirname "$cluster_config")"

    cat > "$cluster_config" <<EOF
apiVersion: k3d.io/v1alpha5
kind: Simple
metadata:
  name: ${name}
servers: ${servers}
agents: ${agents}
network: dt-${name}
subnet: ${network_cidr}
options:
  k3s:
    extraArgs:
      - arg: "--cluster-cidr=${pod_cidr}"
        nodeFilters:
          - server:*
      - arg: "--service-cidr=${service_cidr}"
        nodeFilters:
          - server:*
  kubeconfig:
    updateDefaultKubeconfig: false
    switchCurrentContext: false
labels:
  - key: "dt.cluster.type"
    value: "${type}"
  - key: "dt.cluster.name"
    value: "${name}"
EOF

    k3d cluster create "$name" --config "$cluster_config" --wait --timeout 180s
    k3d kubeconfig merge "$name" --kubeconfig-merge-default || true

    export KUBECONFIG="$(k3d kubeconfig write "$name")"
    kubectl wait node --all --for=condition=Ready --timeout=180s >/dev/null 2>&1 || true
    annotate_nodes "$name" "$type" "$cluster_id" "$pod_cidr" "$service_cidr" "$network_cidr" "$accelerator_hint"
    unset KUBECONFIG

    echo "    ✓ Cluster '$name' created"
}

if [ "$CLEAN" = true ]; then
    echo "Cleaning up existing clusters..."
    for spec in "${CLUSTER_SPECS[@]}"; do
        IFS='|' read -r cluster_name _ <<< "$spec"
        if k3d cluster list | grep -q "^${cluster_name}[[:space:]]"; then
            echo "  Deleting cluster: $cluster_name"
            k3d cluster delete "$cluster_name" || true
        fi
    done
    echo "✓ Cleanup complete"
    echo ""
fi

# Create clusters in parallel
echo "Creating clusters..."
PIDS=()
for spec in "${CLUSTER_SPECS[@]}"; do
    IFS='|' read -r cluster_name cluster_type servers agents cluster_id network_cidr pod_cidr service_cidr accelerator_hint <<< "$spec"
    (
        create_cluster "$cluster_name" "$cluster_type" "$servers" "$agents" "$cluster_id" "$network_cidr" "$pod_cidr" "$service_cidr" "$accelerator_hint"
    ) &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "=== Installing Metrics Server ==="

# Install metrics-server in each cluster
for spec in "${CLUSTER_SPECS[@]}"; do
    IFS='|' read -r cluster_name _ <<< "$spec"
    if [ "$SKIP_METRICS" = true ]; then
        echo "  Skipping metrics-server installation (--skip-metrics)"
        break
    fi

    echo "  Installing metrics-server in '$cluster_name'..."

    export KUBECONFIG="$(k3d kubeconfig write "$cluster_name")"
    kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml || true
    kubectl patch deployment metrics-server -n kube-system --type=json \
        -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]' || true
    kubectl wait --for=condition=available deployment/metrics-server -n kube-system --timeout=120s || true
    unset KUBECONFIG

    echo "    ✓ Metrics-server installed in '$cluster_name'"
done

unset KUBECONFIG

echo ""
echo "=== Cluster Status ==="
for spec in "${CLUSTER_SPECS[@]}"; do
    IFS='|' read -r cluster_name _ <<< "$spec"
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
for spec in "${CLUSTER_SPECS[@]}"; do
    IFS='|' read -r cluster_name _ <<< "$spec"
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
