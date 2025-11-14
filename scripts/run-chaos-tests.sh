#!/bin/bash
set -euo pipefail

# Run chaos engineering tests
# Usage: ./scripts/run-chaos-tests.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Running Chaos Engineering Tests ==="
echo ""

# Ensure namespace exists
kubectl create namespace dt-fabric --dry-run=client -o yaml | kubectl apply -f -

# Test 1: Network Partition
echo "Test 1: Network Partition (zone-a <-> zone-b)"
echo "Applying network partition chaos..."
kubectl apply -f "$PROJECT_ROOT/chaos/scenarios/network_partition.yaml"
echo "✓ Network partition applied"
echo "Waiting 10 seconds..."
sleep 10
echo "Cleaning up network partition..."
kubectl delete -f "$PROJECT_ROOT/chaos/scenarios/network_partition.yaml" --ignore-not-found=true
sleep 5
echo "✓ Network partition test complete"
echo ""

# Test 2: Zone Blackout
echo "Test 2: Zone Blackout (zone-a pod failures)"
echo "Applying zone blackout chaos..."
kubectl apply -f "$PROJECT_ROOT/chaos/scenarios/zone_blackout.yaml"
echo "✓ Zone blackout applied"
echo "Waiting 10 seconds..."
sleep 10
echo "Cleaning up zone blackout..."
kubectl delete -f "$PROJECT_ROOT/chaos/scenarios/zone_blackout.yaml" --ignore-not-found=true
sleep 5
echo "✓ Zone blackout test complete"
echo ""

echo "=== Chaos Tests Complete ==="
echo ""
echo "Check DT API logs for resilience behavior:"
echo "  kubectl logs -l app=dt-api -n dt-fabric --tail=50"

