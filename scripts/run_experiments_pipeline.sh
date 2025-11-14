#!/bin/bash
# Complete experiment pipeline: ensure port-forward, run experiments 5 times, generate report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DT_URL="${1:-http://127.0.0.1:8080}"
N_RUNS="${2:-5}"

echo "=========================================="
echo "DIGITAL TWIN EXPERIMENT PIPELINE"
echo "=========================================="
echo "Target: $DT_URL"
echo "Runs per experiment: $N_RUNS"
echo ""

# Step 1: Ensure port-forward is active
echo "Step 1: Ensuring port-forward is active..."
"$SCRIPT_DIR/ensure-port-forward.sh" || {
    echo "Starting port-forward daemon..."
    "$SCRIPT_DIR/port-forward-daemon.sh" > /tmp/pf-daemon.log 2>&1 &
    sleep 5
    "$SCRIPT_DIR/ensure-port-forward.sh"
}
echo "✓ Port-forward active"
echo ""

# Step 2: Verify API is accessible
echo "Step 2: Verifying API accessibility..."
if ! curl -s --max-time 5 "$DT_URL/snapshot" > /dev/null; then
    echo "✗ API not accessible at $DT_URL"
    echo "  Waiting 10 seconds and retrying..."
    sleep 10
    if ! curl -s --max-time 5 "$DT_URL/snapshot" > /dev/null; then
        echo "✗ API still not accessible. Please check:"
        echo "  1. DT API pod is running: kubectl get pods -n dt-fabric -l app=dt-api"
        echo "  2. Port-forward is active: ./scripts/port-forward-daemon.sh"
        exit 1
    fi
fi
echo "✓ API accessible"
echo ""

# Step 3: Run experiments with metrics collection
echo "Step 3: Running experiments ($N_RUNS times each)..."
echo "This may take several minutes..."
echo ""

cd "$PROJECT_ROOT"
python3 experiments/collect_metrics.py "$DT_URL" "$N_RUNS"

# Step 4: Report completion
echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Results are in: reports/experiments/"
echo ""
ls -lh "$PROJECT_ROOT/reports/experiments/" | tail -5

