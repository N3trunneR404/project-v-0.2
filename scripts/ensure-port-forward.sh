#!/bin/bash
# Ensure port-forward is active - quick check and start if needed
# This is a lightweight version for one-time checks

set -euo pipefail

NAMESPACE="dt-fabric"
SERVICE="dt-api"
LOCAL_PORT="8080"
REMOTE_PORT="8080"
PID_FILE="/tmp/dt-api-port-forward.pid"

# Check if port-forward is already running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        # Check if port is actually accessible
        if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/$LOCAL_PORT" 2>/dev/null; then
            echo "✓ Port-forward already active (PID: $PID)"
            exit 0
        else
            # PID exists but port not accessible, clean up
            kill "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
        fi
    else
        rm -f "$PID_FILE"
    fi
fi

# Kill any stale port-forwards
pkill -f "kubectl port-forward.*$SERVICE" || true
sleep 1

# Start new port-forward
echo "Port-forward not active, setting up (attempt 1/5)..."
for i in {1..5}; do
    kubectl port-forward -n "$NAMESPACE" "svc/$SERVICE" "$LOCAL_PORT:$REMOTE_PORT" > /tmp/pf-simple.log 2>&1 &
    PF_PID=$!
    echo "$PF_PID" > "$PID_FILE"
    
    # Wait and verify
    sleep 3
    if kill -0 "$PF_PID" 2>/dev/null; then
        if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/$LOCAL_PORT" 2>/dev/null; then
            echo "✓ Port-forward established successfully (PID: $PF_PID)"
            exit 0
        fi
    fi
    
    # Failed, try again
    kill "$PF_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    if [[ $i -lt 5 ]]; then
        echo "Attempt $i failed, retrying..."
        sleep 2
    fi
done

echo "✗ Failed to establish port-forward after 5 attempts"
exit 1
