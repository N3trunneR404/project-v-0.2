#!/bin/bash
# Robust port-forward daemon that auto-restarts on failure
# Usage: ./scripts/port-forward-daemon.sh [stop]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="/tmp/dt-api-port-forward.pid"
LOG_FILE="/tmp/dt-api-port-forward.log"
NAMESPACE="dt-fabric"
SERVICE="dt-api"
LOCAL_PORT="8080"
REMOTE_PORT="8080"
MAX_RETRIES=100
RETRY_DELAY=2

# Stop existing port-forward if requested
if [[ "${1:-}" == "stop" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping port-forward (PID: $PID)..."
            kill "$PID" 2>/dev/null || true
            sleep 1
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    pkill -f "kubectl port-forward.*$SERVICE" || true
    echo "Port-forward stopped"
    exit 0
fi

# Check if already running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Port-forward already running (PID: $PID)"
        echo "Use './scripts/port-forward-daemon.sh stop' to stop it"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Function to start port-forward
start_port_forward() {
    local attempt=$1
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting port-forward (attempt $attempt)..." >> "$LOG_FILE"
    
    # Kill any existing port-forwards for this service
    pkill -f "kubectl port-forward.*$SERVICE" || true
    sleep 1
    
    # Start new port-forward in background
    kubectl port-forward -n "$NAMESPACE" "svc/$SERVICE" "$LOCAL_PORT:$REMOTE_PORT" >> "$LOG_FILE" 2>&1 &
    local pf_pid=$!
    
    # Wait a moment and check if it's still running
    sleep 2
    if kill -0 "$pf_pid" 2>/dev/null; then
        echo "$pf_pid" > "$PID_FILE"
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Port-forward started (PID: $pf_pid)" >> "$LOG_FILE"
        return 0
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Port-forward failed to start" >> "$LOG_FILE"
        return 1
    fi
}

# Function to check if port-forward is working
check_port_forward() {
    local pid=$1
    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi
    
    # Try to connect to the port
    if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/$LOCAL_PORT" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Main loop
echo "Starting port-forward daemon..."
echo "Logs: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Stop with: ./scripts/port-forward-daemon.sh stop"
echo ""

attempt=1
while [[ $attempt -le $MAX_RETRIES ]]; do
    if start_port_forward $attempt; then
        PF_PID=$(cat "$PID_FILE")
        
        # Monitor and restart if needed
        while [[ $attempt -le $MAX_RETRIES ]]; do
            sleep 5
            
            if ! check_port_forward "$PF_PID"; then
                echo "[$(date +'%Y-%m-%d %H:%M:%S')] Port-forward died, restarting..." >> "$LOG_FILE"
                rm -f "$PID_FILE"
                attempt=$((attempt + 1))
                break
            fi
        done
        
        if [[ $attempt -gt $MAX_RETRIES ]]; then
            break
        fi
    else
        attempt=$((attempt + 1))
        if [[ $attempt -le $MAX_RETRIES ]]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Retrying in $RETRY_DELAY seconds..." >> "$LOG_FILE"
            sleep "$RETRY_DELAY"
        fi
    fi
done

if [[ $attempt -gt $MAX_RETRIES ]]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Max retries reached, giving up" >> "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

