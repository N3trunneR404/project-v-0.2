#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/reports/experiments"
LOGS_DIR="$PROJECT_ROOT/logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

echo "=== Running All Experiments ==="
echo "Results will be saved to: $RESULTS_DIR"
echo "Logs will be saved to: $LOGS_DIR"
echo ""

# Ensure port-forward is running
if ! curl -s http://127.0.0.1:8080/plan > /dev/null 2>&1; then
    echo "⚠ Port-forward not active. Setting up..."
    DT_API_POD=$(kubectl get pods -n dt-fabric -l app=dt-api -o jsonpath='{.items[0].metadata.name}')
    kubectl port-forward -n dt-fabric "pod/$DT_API_POD" 8080:8080 > "$LOGS_DIR/port-forward.log" 2>&1 &
    sleep 5
    if ! curl -s http://127.0.0.1:8080/plan > /dev/null 2>&1; then
        echo "❌ Failed to establish port-forward"
        exit 1
    fi
    echo "✓ Port-forward established"
fi

# Run experiment suite
echo "Running experiment suite..."
cd "$PROJECT_ROOT"
python3 experiments/run_suite.py 2>&1 | tee "$LOGS_DIR/experiments-run-$(date +%Y%m%d-%H%M%S).log"

echo ""
echo "=== Experiments Complete ==="
echo "Check results in: $RESULTS_DIR"
echo "Check logs in: $LOGS_DIR"

