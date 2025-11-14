# Port-Forward Fix

## Problem
Port-forward connections to the DT API were unstable:
- Connections would die after a short time
- No automatic restart mechanism
- Experiments would fail with connection errors

## Solution

### 1. Port-Forward Daemon (`scripts/port-forward-daemon.sh`)
A robust daemon that:
- Automatically restarts port-forward on failure
- Monitors connection health every 5 seconds
- Retries up to 100 times before giving up
- Logs all activity to `/tmp/dt-api-port-forward.log`
- Stores PID in `/tmp/dt-api-port-forward.pid` for easy management

**Usage:**
```bash
# Start daemon (runs in background)
./scripts/port-forward-daemon.sh

# Stop daemon
./scripts/port-forward-daemon.sh stop

# Check status
cat /tmp/dt-api-port-forward.pid
tail -f /tmp/dt-api-port-forward.log
```

### 2. Quick Check Script (`scripts/ensure-port-forward.sh`)
Lightweight script for one-time checks:
- Verifies if port-forward is active
- Starts one if needed
- Quick 5-attempt retry logic
- Useful for scripts that need a quick check

**Usage:**
```bash
./scripts/ensure-port-forward.sh
```

## Features

1. **Auto-Restart**: Automatically restarts failed connections
2. **Health Monitoring**: Checks connection every 5 seconds
3. **Port Verification**: Verifies TCP connection before declaring success
4. **Clean Shutdown**: Properly kills processes and cleans up PID files
5. **Logging**: All activity logged for debugging

## Testing

```bash
# Start daemon
./scripts/port-forward-daemon.sh

# Test connection
curl http://127.0.0.1:8080/snapshot

# Run experiments (daemon keeps connection alive)
python3 experiments/run_suite.py
```

## Status

✅ **Fixed**: Port-forward now stable and auto-restarting
- Daemon keeps connection alive
- Automatic recovery from failures
- Ready for long-running experiments

